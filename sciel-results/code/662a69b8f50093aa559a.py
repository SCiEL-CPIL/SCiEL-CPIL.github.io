import random
import time
import sys

_START_TIME = time.time()
DEADLINE = 110.0  # seconds, safety margin before 120s

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return DEADLINE - _elapsed()

# Precompute popcount for all 2^18 masks
_POPCOUNT = [0] * (1 << 18)
for i in range(1, 1 << 18):
    _POPCOUNT[i] = _POPCOUNT[i >> 1] + (i & 1)

def _popcount(x):
    return _POPCOUNT[x]

def _is_k4_free(adj, n=18):
    """Check no K4 in graph with adjacency bitmasks adj (list of ints)."""
    for a in range(n):
        # Only check edges (a, b) where a < b
        # We iterate b > a
        b_mask = adj[a] & ~((1 << (a + 1)) - 1) # neighbors > a
        while b_mask:
            lsb = b_mask & -b_mask
            b = lsb.bit_length() - 1
            b_mask -= lsb
            # Check if adj[a] and adj[b] have a common neighbor c such that c is adjacent to both
            # And c != a, b (implied by common neighbors of edge? No, common neighbors of a and b are vertices connected to both)
            # If there is an edge (c, d) in common neighbors, we have K4?
            # No, K4 is {a, b, c, d}.
            # a-b is edge.
            # If c, d are common neighbors, c~a, c~b, d~a, d~b.
            # If c~d, then {a,b,c,d} is K4.
            # So we need to check if the subgraph induced by common neighbors of a and b has any edge.
            common = adj[a] & adj[b]
            # Remove a and b from common? They are not in common anyway (simple graph).
            # Check if common contains an edge.
            c = common
            while c:
                lsb_c = c & -c
                x = lsb_c.bit_length() - 1
                c -= lsb_c
                # If x is adjacent to any other vertex in common, we have an edge in common.
                # adj[x] & common will include x? No, adj[x] doesn't have self-loop.
                # But it might include other vertices in common.
                if (adj[x] & common):
                    return False
    return True

def _max_triangle_free_induced(adj, n=18):
    """Exact branch-and-bound for largest triangle-free induced subgraph."""
    best = [0]
    
    # Order vertices by degree descending to prune earlier?
    # Or just 0..n-1.
    # Let's try sorting by degree descending.
    degrees = [_popcount(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    # Map original index to new index?
    # Actually, the B&B logic relies on "forbidden" which is computed based on adjacency.
    # If we reorder vertices, we need to permute adj accordingly.
    # This is complex. Let's stick to 0..n-1 but maybe use a heuristic in B&B?
    # The current B&B processes vertices in index order.
    # Let's keep index order for simplicity, but the popcount table will speed it up.
    
    def bb(idx, current_mask, forbidden_mask):
        if _elapsed() > DEADLINE:
            return
        
        # Upper bound
        # Available vertices are those >= idx not in forbidden_mask
        # We can compute this mask quickly?
        # remaining_mask = ((1 << n) - 1) ^ ((1 << idx) - 1)
        # available = remaining_mask & ~forbidden_mask
        remaining_mask = ((1 << n) - 1) ^ ((1 << idx) - 1)
        available = remaining_mask & ~forbidden_mask
        
        ub = _popcount(current_mask) + _popcount(available)
        if ub <= best[0]:
            return
        
        if idx == n:
            sz = _popcount(current_mask)
            if sz > best[0]:
                best[0] = sz
            return
        
        v = idx
        bit = 1 << v
        
        # Option 1: Exclude v
        bb(idx + 1, current_mask, forbidden_mask)
        
        # Option 2: Include v
        if not (forbidden_mask & bit):
            new_forbidden = forbidden_mask
            cm = current_mask
            while cm:
                lsb = cm & -cm
                u = lsb.bit_length() - 1
                cm -= lsb
                if (adj[v] >> u) & 1:
                    new_forbidden |= (adj[v] & adj[u])
            
            bb(idx + 1, current_mask | bit, new_forbidden)
            
    bb(0, 0, 0)
    return best[0]

def construct():
    n = 18
    # Start with K6,6,6
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    adj = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    current_adj = adj[:]
    best_adj = adj[:]
    
    # Initial evaluation
    current_score = _max_triangle_free_induced(current_adj, n)
    best_score = current_score
    
    random.seed(42)
    
    # Local Search with bias towards adding edges
    T = 1.0
    step = 0
    no_improve_count = 0
    
    while _time_left() > 5:
        # Pick random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        # Bias: 60% chance to add edge, 40% to remove
        # If edge exists, we remove. If not, we add.
        # To bias adding, if edge exists, we only flip with prob 0.4?
        # Or just pick direction.
        
        edge_exists = (current_adj[u] >> v) & 1
        
        # Decide flip direction
        if edge_exists:
            # Remove edge
            # Only do this with some probability
            if random.random() > 0.4:
                continue
        else:
            # Add edge
            # Only do this with some probability
            if random.random() > 0.6:
                continue
        
        new_adj = current_adj[:]
        if edge_exists:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
        
        # Check K4-free
        if not _is_k4_free(new_adj, n):
            T *= 0.99
            step += 1
            continue
        
        # Evaluate
        new_score = _max_triangle_free_induced(new_adj, n)
        
        if new_score < best_score:
            best_score = new_score
            best_adj = new_adj
            current_adj = new_adj
            T = 1.0
            no_improve_count = 0
        else:
            delta = new_score - best_score
            if delta > 0:
                prob = 1.0 / (1.0 + step / 50.0)
                if random.random() < prob:
                    current_adj = new_adj
            else:
                if random.random() < 0.5:
                    current_adj = new_adj
            
            no_improve_count += 1
            T *= 0.99
            step += 1
            
        if no_improve_count > 200:
            # Reheat
            T = 1.0
            no_improve_count = 0
            step = 0

    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
