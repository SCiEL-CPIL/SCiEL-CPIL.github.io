import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def get_pairs(n):
        return [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def pair_index(n, u, v):
        # u < v
        # Index in lexicographic order
        # Number of pairs with first element < u is u*(n-1) - u*(u-1)/2 ?
        # Let's just precompute a dictionary for speed
        pass

    def count_violations(colors, n):
        pairs = get_pairs(n)
        N = len(pairs)
        # Map pair to index
        pair_to_idx = {p: i for i, p in enumerate(pairs)}
        
        adj_0 = [set() for _ in range(N)]
        adj_1 = [set() for _ in range(N)]
        
        # Precompute disjointness and mono-color
        # For each pair i, j
        for i in range(N):
            u1, u2 = pairs[i]
            for j in range(i+1, N):
                v1, v2 = pairs[j]
                # Check disjoint
                if u1 in (v1, v2) or u2 in (v1, v2):
                    continue
                
                # Get colors of 4 edges
                # Edges: (u1,v1), (u1,v2), (u2,v1), (u2,v2)
                # Ensure correct order for index lookup
                def get_c(a, b):
                    if a > b: a, b = b, a
                    return colors[pair_to_idx[(a,b)]]
                
                c1 = get_c(u1, v1)
                c2 = get_c(u1, v2)
                c3 = get_c(u2, v1)
                c4 = get_c(u2, v2)
                
                if c1 == c2 == c3 == c4:
                    if c1 == 0:
                        adj_0[i].add(j)
                        adj_0[j].add(i)
                    else:
                        adj_1[i].add(j)
                        adj_1[j].add(i)
        
        list_v0 = []
        list_v1 = []
        
        # Find triangles in adj_0
        for i in range(N):
            for j in list(adj_0[i]):
                if j > i:
                    common = adj_0[i] & adj_0[j]
                    for k in common:
                        if k > j:
                            list_v0.append((i, j, k))
                            
        # Find triangles in adj_1
        for i in range(N):
            for j in list(adj_1[i]):
                if j > i:
                    common = adj_1[i] & adj_1[j]
                    for k in common:
                        if k > j:
                            list_v1.append((i, j, k))
                            
        return list_v0, list_v1, pairs, pair_to_idx

    def fix_violation(colors, n, violation, pairs, pair_to_idx, color):
        # violation is (i, j, k) indices of pairs
        # The edges involved are between pairs i, j, k
        # We pick a random edge from the 12 edges and flip it
        p_i = pairs[violation[0]]
        p_j = pairs[violation[1]]
        p_k = pairs[violation[2]]
        
        edges = []
        # Edges between p_i and p_j
        for u in p_i:
            for v in p_j:
                edges.append((u, v))
        # Edges between p_i and p_k
        for u in p_i:
            for v in p_k:
                edges.append((u, v))
        # Edges between p_j and p_k
        for u in p_j:
            for v in p_k:
                edges.append((u, v))
                
        # Pick random edge
        u, v = random.choice(edges)
        if u > v: u, v = v, u
        idx = pair_to_idx[(u, v)]
        colors[idx] = 1 - colors[idx]
        return colors

    best_n = 0
    best_colors = None
    
    # Try n=30 first
    for n in [30, 29, 28]:
        if time.time() > deadline - 5:
            break
            
        colors = [random.randint(0, 1) for _ in range(len(get_pairs(n)))]
        
        # Local search
        while time.time() < deadline - 5:
            list_v0, list_v1, pairs, pair_to_idx = count_violations(colors, n)
            
            if not list_v0 and not list_v1:
                best_n = n
                best_colors = colors
                break
                
            # Pick a random violation
            if list_v0 and (not list_v1 or random.random() < 0.5):
                v = random.choice(list_v0)
                colors = fix_violation(colors, n, v, pairs, pair_to_idx, 0)
            else:
                v = random.choice(list_v1)
                colors = fix_violation(colors, n, v, pairs, pair_to_idx, 1)
                
        if best_n == n:
            break # Found valid coloring for this n
            
    if best_n > 0:
        return {'n': best_n, 'colors': best_colors}
    else:
        # Fallback: Return a valid small coloring (e.g., n=1 or 2) to avoid error
        # But we should try to return something valid.
        # A random coloring of n=1 is always valid.
        return {'n': 1, 'colors': []}

# EVOLVE-BLOCK-END
