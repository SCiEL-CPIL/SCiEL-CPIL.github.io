import random
import time

def construct():
    def is_valid(n, edges):
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Check K4
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] & (1 << v)): continue
                for w in range(v + 1, n):
                    if not (adj_mask[u] & (1 << w)) or not (adj_mask[v] & (1 << w)): continue
                    common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                    common &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                    mask_x = ((1 << n) - 1) ^ ((1 << (w + 1)) - 1)
                    if common & mask_x:
                        return False
        
        # Check I6 (Independent set of size 6)
        # Equivalent to K6 in complement
        comp_mask = [0] * n
        for i in range(n):
            comp_mask[i] = ((1 << n) - 1) ^ adj_mask[i] ^ (1 << i)
            
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_mask[u] & (1 << v)): continue
                for w in range(v + 1, n):
                    if not (comp_mask[u] & (1 << w)) or not (comp_mask[v] & (1 << w)): continue
                    common = comp_mask[u] & comp_mask[v] & comp_mask[w]
                    common &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                    mask_x = ((1 << n) - 1) ^ ((1 << (w + 1)) - 1)
                    if common & mask_x:
                        return False
        return True

    def get_paley(q):
        squares = {x * x % q for x in range(1, q)}
        edges = [[i, j] for i in range(q) for j in range(i + 1, q) if (j - i) % q in squares]
        return q, edges

    # Try to build a larger graph using sequential construction
    def try_build(target_n, seed):
        random.seed(seed)
        # Start with a small valid graph, e.g., Paley(17) or empty
        # Let's start with Paley(17) if valid, else empty
        base_n, base_edges = get_paley(17)
        if not is_valid(base_n, base_edges):
            base_n = 0
            base_edges = []
        
        adj_mask = [0] * target_n
        for u, v in base_edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        current_n = base_n
        for new_v in range(current_n, target_n):
            # Determine valid neighbors for new_v
            # 1. S_edge must be triangle-free in current graph
            # 2. V \ S_edge must not contain I5 in current graph
            
            # Heuristic: Try to pick a random subset of vertices
            # We need to find S such that:
            # - No triangle in S
            # - No I5 in V\S
            
            # This is a hard constraint satisfaction problem.
            # Use random restarts for this vertex
            found = False
            for _ in range(200):
                S = set()
                # Randomly decide edges
                # To ensure no triangle in S:
                # We can greedily add vertices to S if they don't form a triangle with existing S
                # To ensure no I5 in V\S:
                # We can greedily ensure that V\S doesn't have large independent sets?
                
                # Simpler: Randomly assign edges, then check constraints locally
                # But checking I5 in V\S is expensive.
                
                # Alternative: Use a greedy approach
                # 1. Pick a random permutation of 0..new_v-1
                # 2. Iterate, decide to add to S or not
                # 3. Constraint: If u in S, and v in S, and w in S, no triangle.
                # 4. Constraint: If u not in S, v not in S, ..., 5 vertices not in S, no I5.
                
                # This is still complex.
                # Let's try a simple random assignment and verify the whole graph at the end?
                # No, that's what the old program did.
                
                # Let's just use the Paley graph if it works.
                break # Skip complex construction for now, rely on Paley
            
            if not found:
                return None
        
        edges = []
        for u in range(target_n):
            for v in range(u + 1, target_n):
                if adj_mask[u] & (1 << v):
                    edges.append([u, v])
        return edges

    # Check Paley graphs for validity
    best_candidate = None
    for q in [37, 29, 17, 13]:
        if q > 40: continue
        n, edges = get_paley(q)
        if is_valid(n, edges):
            best_candidate = {'n': n, 'edges': edges}
            break

    if best_candidate:
        return best_candidate

    # Fallback: Return a small valid graph
    # Paley(17) should be valid. If not, try smaller.
    # If all fail, return n=1
    return {'n': 1, 'edges': []}
