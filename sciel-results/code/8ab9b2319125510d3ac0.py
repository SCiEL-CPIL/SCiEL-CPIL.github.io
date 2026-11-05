import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds safety margin
    
    # Fallback solution n=11
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    
    best_n = n_fallback
    best_colors = colors_fallback
    
    # Try to find solutions for larger n
    for n in range(16, 30):
        if time.time() > deadline:
            break
            
        # Initialize random coloring
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Helper to convert colors list to adjacency matrices
        def get_matrices(colors, n):
            mats = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    mats[c][i, j] = 1
                    mats[c][j, i] = 1
                    idx += 1
            return mats
        
        # Helper to count C4s
        def count_c4s(mats):
            total = 0
            for m in mats:
                # m is symmetric 0/1 matrix
                # m^2 gives number of common neighbors
                # We want to count pairs (u,v) with m[u,v]=1 and (m^2)[u,v] >= 2
                # Actually, if m[u,v]=1, a C4 is formed if there are 2 other paths of length 2.
                # Number of C4s is 1/8 * sum_{u,v} ( (m^2)_{uv} choose 2 )? 
                # Let's use a simpler metric: sum of (m^2)_{uv} for u!=v, u!=i, v!=j?
                # Standard metric: sum over all u,v of (m^2)_{uv}. 
                # Minimizing this helps.
                # Better: count number of "bad" edges.
                m2 = m @ m
                # For each edge (i,j), if m2[i,j] >= 2, it's part of a C4.
                # Sum of (m2[i,j] - 1) for all edges?
                # Let's just sum m2[i,j] for all i<j where m[i,j]=1.
                # This penalizes high common neighbor counts.
                edge_mask = np.triu(m, 1)
                total += np.sum(m2[edge_mask])
            return total

        # Local Search
        mats = get_matrices(colors, n)
        current_cost = count_c4s(mats)
        
        # Simple hill climbing with restarts
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if current_cost == 0:
                # Found valid coloring
                best_n = n
                best_colors = colors[:]
                break
                
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            new_c = random.randint(0, 4)
            if old_c == new_c:
                continue
                
            # Update matrices
            # Find i,j from e_idx
            # e_idx = i*n - (i+1)/2 + (j-i-1) ... inverse is slow.
            # Precompute edge endpoints
            # For speed, we can just recompute cost if we update matrices efficiently.
            # But updating matrices and recomputing full cost is O(n^2).
            # n=16, n^2=256. 5000 * 256 * 5 ops is fine.
            
            # Find i, j
            # k = e_idx
            # i = floor((sqrt(8*k + 1) - 1) / 2)
            # j = k - i*(n - 1) + i*(i+1)/2 ... 
            # Let's just use a precomputed list of edges
            # To avoid overhead, we'll just recompute cost from scratch for simplicity in this draft, 
            # but optimize if needed.
            
            # Actually, let's just recompute.
            colors[e_idx] = new_c
            mats_new = get_matrices(colors, n)
            new_cost = count_c4s(mats_new)
            
            if new_cost < current_cost:
                current_cost = new_cost
                mats = mats_new
            else:
                colors[e_idx] = old_c
        
        if current_cost == 0:
            best_n = n
            best_colors = colors[:]
            # Continue to try for larger n? 
            # The problem asks to maximize n. 
            # If we found 16, we should try 17, etc.
            # But if 16 is hard, 17 is harder.
            # We'll continue the loop.

    return {'n': best_n, 'colors': best_colors}
