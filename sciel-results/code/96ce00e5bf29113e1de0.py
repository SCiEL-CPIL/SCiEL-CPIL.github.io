import random
import time
import sys

_START_TIME = time.time()
DEADLINE = 105.0  # seconds, safety margin before 120s limit

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _is_k4_free(adj, n=18):
    """Check no K4 in graph with adjacency bitmasks adj (list of ints)."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            # Check if common contains an edge (i.e., two vertices in common are adjacent)
            c = common
            while c:
                lsb = c & -c
                x = lsb.bit_length() - 1
                c -= lsb
                if (adj[x] & common) != 0:
                    return False
    return True

def _max_triangle_free_induced(adj, n=18):
    """Exact branch-and-bound for largest triangle-free induced subgraph."""
    best = [0]
    
    def popcount(x):
        return bin(x).count('1')
    
    # Order vertices by degree descending to improve pruning
    order = list(range(n))
    order.sort(key=lambda v: bin(adj[v]).count('1'), reverse=True)
    
    # Map original vertex to position in order
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    # Precompute common neighbors for edges
    # common[u][v] = adj[u] & adj[v] for edge (u,v)
    
    def bb(idx, current_mask, forbidden_mask):
        if _elapsed() > DEADLINE:
            return
        
        # Upper bound: current size + number of remaining vertices not forbidden
        if idx == n:
            sz = popcount(current_mask)
            if sz > best[0]:
                best[0] = sz
            return
        
        # Count available vertices from idx onwards
        available_count = 0
        for i in range(idx, n):
            v = order[i]
            bit = 1 << v
            if not (forbidden_mask & bit):
                available_count += 1
        
        ub = popcount(current_mask) + available_count
        if ub <= best[0]:
            return
        
        v = order[idx]
        bit = 1 << v
        
        # Option 1: Exclude v
        bb(idx + 1, current_mask, forbidden_mask)
        
        # Option 2: Include v (if not forbidden)
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
    
    # Local Search with simulated annealing
    T = 1.0
    step = 0
    no_improve_count = 0
    
    while _time_left() > 5:
        # Pick random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        # Flip edge
        new_adj = current_adj[:]
        if (new_adj[u] >> v) & 1:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
        
        # Check K4-free
        if not _is_k4_free(new_adj, n):
            continue
        
        new_score = _max_triangle_free_induced(new_adj, n)
        
        if new_score < best_score:
            best_score = new_score
            best_adj = new_adj
            current_adj = new_adj
            no_improve_count = 0
            T = 1.0
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = 1.0 / (1.0 + step / 100.0)
                if random.random() < prob:
                    current_adj = new_adj
            else:
                current_adj = new_adj
            no_improve_count += 1
        
        step += 1
        T *= 0.999
        
        if best_score <= 9:
            break
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
