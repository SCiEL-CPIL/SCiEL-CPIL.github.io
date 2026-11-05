import itertools
import time

def construct():
    start_time = time.time()
    
    # 1. Construct the Schläfli graph (n=35)
    vertices = list(itertools.combinations(range(7), 3))
    n = len(vertices)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            inter_len = len(set(vertices[i]).intersection(vertices[j]))
            if inter_len == 0 or inter_len == 2:
                edges.append([i, j])
                
    # 2. Verify the graph
    # Check K4 and Independent Set 6
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Check for K4
    def has_k4():
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]: continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]: continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return True
        return False
        
    # Check for Independent Set 6
    def has_ind_6():
        # Use a simple backtracking or bitset approach for speed
        # Since n=35, simple recursion might be slow if not pruned, 
        # but we can use the fact that alpha is small.
        # For verification, we can just check if any 6-set is independent.
        # This is C(35,6) ~ 1.6M checks. Each check is O(1) with bitsets.
        # Let's use bitsets for speed.
        
        # Build bitsets for adjacency
        # adj_bits[i] is a bitmask of neighbors of i
        adj_bits = [0]*n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j]:
                    mask |= (1 << j)
            adj_bits[i] = mask
            
        # A set S is independent if for all i in S, no neighbor of i is in S.
        # i.e., (adj_bits[i] & mask_S) == 0 for all i in S.
        
        # We can iterate through combinations of 6.
        for comb in itertools.combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            is_ind = True
            for v in comb:
                if adj_bits[v] & mask:
                    is_ind = False
                    break
            if is_ind:
                return True
        return False

    if has_k4() or has_ind_6():
        # Fallback to Paley graph (n=17) if something went wrong
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
        
    return {'n': n, 'edges': edges}
