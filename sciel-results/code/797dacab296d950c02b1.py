import itertools
import random
import time
import math

def construct():
    n = 18
    
    # Precompute common neighbors for fast triangle check
    # common[u][v] will be a bitmask of vertices connected to both u and v
    # We will update this dynamically or recompute?
    # Recomputing is O(n^2) which is 324 ops. Very fast.
    
    def is_triangle_free_mask(mask, adj):
        # Check if the subgraph induced by 'mask' is triangle-free
        # Iterate through all pairs (u, v) in mask
        # If edge (u,v) exists, check if any common neighbor is in mask
        u = 0
        while u < n:
            if (mask >> u) & 1:
                v = u + 1
                # Only check edges (u, v) where u < v
                while v < n:
                    if (mask >> v) & 1:
                        if (adj[u] >> v) & 1:
                            # Edge (u,v) exists in the subgraph
                            # Check for common neighbor in mask
                            common = adj[u] & adj[v]
                            if (common & mask) != 0:
                                return False
                    v += 1
            u += 1
        return True

    def get_score_exact(adj):
        # Find largest s such that there exists a subset of size s that is triangle-free
        # Iterate s from n down to 1
        for s in range(n, 0, -1):
            for comb in itertools.combinations(range(n), s):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if is_triangle_free_mask(mask, adj):
                    return s
        return 0

    def check_k4_free(adj, u, v):
        # Check if adding edge (u,v) creates a K4
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        c = common
        while c:
            lsb = c & -c
            k = lsb.bit_length() - 1
            c ^= lsb
            # If k is connected to another common neighbor, K4
            # We need to check if there is any edge between k and any other node in 'common'
            # Actually, if k is in common, and there is any other node m in common such that (k,m) is an edge,
            # then {u, v, k, m} is a K4.
            # So we check if adj[k] & common has any bit set other than k itself?
            # adj[k] & common gives neighbors of k that are also neighbors of u and v.
            # If this is non-zero, it means k is connected to some m in common.
            # Since m is in common, m is connected to u and v.
            # So u-v, u-k, u-m, v-k, v-m, k-m are all edges.
            # Thus K4.
            if (adj[k] & common) != 0:
                return False
        return True

    # Initialize
    adj = [0] * n
    random.seed(42)
    
    # Greedy Initialization: 
    # Try to add edges that create triangles to reduce the score.
    # Random order, but accept if K4-free.
    # To bias towards triangles, we can shuffle edges but prioritize those with high common neighbor count?
    # Let's just do random shuffle for simplicity, then local search will fix it.
    edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges_list)
    
    for u, v in edges_list:
        if check_k4_free(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    current_score = get_score_exact(adj)
    best_score = current_score
    best_adj = adj[:]
    
    start_time = time.time()
    timeout = 115
    
    T = 1.0
    cooling = 0.99
    
    while time.time() - start_time < timeout:
        u, v = random.sample(range(n), 2)
        edge_exists = (adj[u] >> v) & 1
        
        new_adj = adj[:]
        if edge_exists:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            if not check_k4_free(adj, u, v):
                continue
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
        # Fast heuristic filter: Count edges in new_adj vs adj
        # More edges usually means smaller TF subgraph, but not always.
        # Let's just compute exact score. It's fast enough for n=18 with bitmask check.
        
        new_score = get_score_exact(new_adj)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            adj = new_adj
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
                
        T *= cooling
        
    adj = best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
