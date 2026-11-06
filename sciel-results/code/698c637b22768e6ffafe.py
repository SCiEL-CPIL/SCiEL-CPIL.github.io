import time
import random
import sys

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _has_c4(adj_masks, n):
    """Check if graph has a C4 using bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_masks[u] & adj_masks[v]
            if bin(common).count('1') > 1:
                return True
    return False

def _max_independent_set_size(adj_masks, n, timeout=2.0):
    """Find size of maximum independent set using branch and bound with bitmasks."""
    start = time.time()
    best = [0]
    
    # Precompute non-neighbor masks (including self? No, independent set excludes self)
    # non_adj[i] has bits set for vertices NOT adjacent to i, and i itself is 0?
    # Let's define avail_mask as bits of vertices that can still be added.
    
    def popcount(x):
        return bin(x).count('1')
        
    def branch(available_mask, current_size):
        if time.time() - start > timeout:
            return
        if available_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + popcount(available_mask) <= best[0]:
            return
        
        # Pick a vertex with minimum degree in the available subgraph
        # To speed up, just pick the first one
        u_bit = available_mask & (-available_mask)
        u = u_bit.bit_length() - 1
        
        # Include u
        # Remove u and its neighbors from available
        new_avail = available_mask & ~(adj_masks[u] | u_bit)
        branch(new_avail, current_size + 1)
        
        # Exclude u
        branch(available_mask & ~u_bit, current_size)
        
    full_mask = (1 << n) - 1
    branch(full_mask, 0)
    return best[0]

def _is_c4_free(adj_masks, n):
    """Check C4-free using bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_masks[u] & adj_masks[v]
            if bin(common).count('1') > 1:
                return False
    return True

def _indep_number(adj_masks, n, timeout=1.0):
    """Compute independence number with branch and bound."""
    return _max_independent_set_size(adj_masks, n, timeout)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)
    
    best_n = 0
    best_edges = []
    
    # We will try to build a graph up to 43 vertices.
    # We start with a small graph and add vertices.
    # To ensure we get a good start, we use a greedy approach with random restarts.
    
    max_attempts = 20
    for attempt in range(max_attempts):
        if _time_remaining() < 10.0:
            break
            
        n = 0
        adj_masks = []
        
        # Start with 0 vertices.
        # Try to add vertices 0 to 42.
        
        success = True
        for new_n in range(1, 44):
            if _time_remaining() < 5.0:
                success = False
                break
                
            # Try to add a vertex
            added = False
            # Limit attempts per vertex to avoid getting stuck
            max_vertex_attempts = 50
            for _ in range(max_vertex_attempts):
                if _time_remaining() < 5.0:
                    break
                    
                # Determine potential neighbors
                # The set S of neighbors must have pairwise disjoint neighborhoods in the current graph.
                # Let's pick a random subset of vertices to be neighbors.
                # To satisfy C4-free, for any u, v in S, N(u) & N(v) == 0.
                
                # Greedy construction of S:
                S = []
                available_vertices = list(range(n))
                random.shuffle(available_vertices)
                
                # We want S to be as large as possible to reduce alpha of the remainder?
                # Actually, we want alpha(G - S) <= 9.
                # Larger S means fewer vertices in G-S, so smaller alpha.
                # So we want a large S.
                
                for u in available_vertices:
                    # Check if u can be added to S
                    # u must not share a neighbor with any vertex in S
                    can_add = True
                    for w in S:
                        if adj_masks[u] & adj_masks[w]:
                            can_add = False
                            break
                    if can_add:
                        S.append(u)
                
                # Check if adding v with neighbors S keeps alpha <= 10
                # New graph has n+1 vertices.
                # v is connected to S.
                # Independent sets either contain v or not.
                # If not: alpha(G) <= 10 (assumed)
                # If yes: 1 + alpha(G - (S U {v})) <= 10 => alpha(G - S) <= 9
                
                # Calculate alpha of G - S
                # G - S is the induced subgraph on V \ S
                remaining = [v for v in range(n) if v not in S]
                if len(remaining) <= 9:
                    alpha_rem = len(remaining)
                else:
                    # Build subgraph masks for remaining
                    # Map remaining vertices to 0..len(remaining)-1
                    rem_idx = {v: i for i, v in enumerate(remaining)}
                    rem_masks = [0] * len(remaining)
                    for i, u in enumerate(remaining):
                        mask = 0
                        for j, w in enumerate(remaining):
                            if i != j and (adj_masks[u] >> w) & 1:
                                mask |= (1 << j)
                        rem_masks[i] = mask
                    
                    alpha_rem = _indep_number(rem_masks, len(remaining), timeout=0.5)
                
                if alpha_rem <= 9:
                    # Valid addition
                    # Create new adj_masks
                    new_adj_masks = adj_masks + [0]
                    v = n
                    for u in S:
                        new_adj_masks[v] |= (1 << u)
                        new_adj_masks[u] |= (1 << v)
                    
                    adj_masks = new_adj_masks
                    n += 1
                    added = True
                    break
            
            if not added:
                success = False
                break
        
        if success and n > best_n:
            best_n = n
            # Convert to edges
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_masks[i] >> j) & 1:
                        edges.append([i, j])
            best_edges = edges
            
            # If we reached 43, we can stop
            if n == 43:
                break

    # If we didn't find any, return a trivial graph? No, must be valid.
    # The loop should find at least something.
    # If best_n is 0, we might have a timeout.
    # Let's ensure we return a valid graph.
    
    if best_n == 0:
        # Fallback: return a small valid graph
        # 1 vertex
        return {'n': 1, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
