import numpy as np
import time
import random

def count_triangles(adj):
    """Count triangles in a graph given by boolean adjacency matrix."""
    # adj is square boolean matrix
    # Number of triangles is trace(A^3) / 6
    # A^3 can be computed as (A @ A) @ A
    # Using integer arithmetic to avoid float issues, though boolean is fine
    a3 = (adj.astype(np.int32) @ adj.astype(np.int32)) @ adj.astype(np.int32)
    return np.trace(a3) // 6

def construct():
    # Seed graph: Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    best_n = q
    best_edges = seed_edges
    
    # Target size to try to find
    target_n = 42
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    # Simulated Annealing
    # We will try to find a valid graph on target_n vertices
    
    for restart in range(100): # Max 100 restarts, but limited by time
        if time.time() > deadline:
            break
            
        # Initialize random graph on target_n vertices
        n = target_n
        # Random graph with p=0.5
        adj = np.random.rand(n, n) < 0.5
        np.fill_diagonal(adj, False)
        # Symmetrize
        adj = np.logical_or(adj, adj.T)
        
        # Calculate initial C5 and I5
        C5 = 0
        I5 = 0
        
        # This initialization is O(n^5) which is slow for n=42 in pure python loops
        # Let's optimize:
        # C5 = sum over edges (u,v) of tri(G[S_uv]) / 10
        # I5 = sum over non-edges (u,v) of tri(~G[T_uv]) / 10
        
        # To speed up, we can just calculate the energy E = C5 + I5
        # and update it.
        
        # Efficient initialization:
        # We can just run the update logic for all edges? No, that's circular.
        # Let's just calculate C5 and I5 directly.
        
        # Vectorized triangle counting for all edges is hard.
        # Let's use a simpler heuristic: just run SA for a few steps and hope it finds 0?
        # No, we need to know if E=0.
        
        # Alternative: Use a faster exact count for n=42.
        # n=42, n^3 = 74000.
        # Number of pairs = 42*41/2 = 861.
        # 861 * 74000 = 63 million operations.
        # In Python loop, this is ~1-2 seconds.
        # In NumPy, we can batch it?
        
        # Let's try to optimize the triangle count for a specific set of vertices S.
        
        def get_delta(u, v, is_add):
            # S: common neighbors of u, v
            # T: common non-neighbors of u, v
            # Note: u, v are not in S or T
            
            # Mask for u, v
            mask_u = adj[u]
            mask_v = adj[v]
            
            # S indices
            s_idx = np.where(np.logical_and(mask_u, mask_v))[0]
            # Remove u and v if they are in s_idx (they shouldn't be, since adj[u,u]=False)
            # But just in case
            s_idx = s_idx[(s_idx != u) & (s_idx != v)]
            
            # T indices: non-neighbors
            # ~mask_u includes u itself (since adj[u,u]=False -> ~False=True)
            # So we must explicitly exclude u and v
            t_mask = np.logical_and(~mask_u, ~mask_v)
            t_mask[u] = False
            t_mask[v] = False
            t_idx = np.where(t_mask)[0]
            
            tri_s = 0
            tri_t = 0
            
            if len(s_idx) >= 3:
                # Build adj for G[S]
                sub_adj = adj[np.ix_(s_idx, s_idx)]
                tri_s = count_triangles(sub_adj)
                
            if len(t_idx) >= 3:
                # Build adj for ~G[T]
                # ~G[T] has edge (i,j) if NOT adj[i,j]
                sub_adj_t = ~adj[np.ix_(t_idx, t_idx)]
                np.fill_diagonal(sub_adj_t, False) # Ensure no self loops in count
                tri_t = count_triangles(sub_adj_t)
                
            if is_add:
                # Add edge (u,v)
                # Creates K5s: tri_s
                # Removes I5s: tri_t
                return tri_s, -tri_t
            else:
                # Remove edge (u,v)
                # Removes K5s: tri_s
                # Creates I5s: tri_t
                return -tri_s, tri_t

        # Initialize C5, I5
        # This is the bottleneck.
        # Let's just estimate or skip exact init and use a "repair" strategy?
        # No, SA needs exact E.
        
        # Let's do the O(n^5) init but optimized.
        C5 = 0
        I5 = 0
        for u in range(n):
            for v in range(u + 1, n):
                dC5, dI5 = get_delta(u, v, True)
                if adj[u, v]:
                    # Edge exists. It contributes dC5 to C5?
                    # No, get_delta(u, v, True) returns change if we ADD edge.
                    # If edge EXISTS, it was added at some point.
                    # The contribution of an existing edge (u,v) to C5 is the number of K5s containing (u,v).
                    # This is exactly tri_s calculated in get_delta(u, v, True).
                    C5 += dC5
                else:
                    # Edge does not exist.
                    # The contribution of a non-edge (u,v) to I5 is the number of I5s containing (u,v).
                    # This is exactly tri_t calculated in get_delta(u, v, False)?
                    # get_delta(u, v, False) returns -tri_s, tri_t.
                    # So tri_t is the contribution to I5.
                    # Let's re-verify.
                    # If we REMOVE edge (u,v), we CREATE tri_t I5s.
                    # So an existing non-edge (u,v) is part of tri_t I5s.
                    I5 += dI5 # dI5 is tri_t in this case?
                    # Wait. get_delta(u, v, False) returns (-tri_s, tri_t).
                    # So dI5 = tri_t. Correct.
                    
        E = C5 + I5
        
        if E == 0:
            # Found valid graph
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            if n > best_n:
                best_n = n
                best_edges = edges
            break # Found one, restart to try for larger? No, target is fixed.
            
        # SA Loop
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        
        while time.time() < deadline and T > min_T:
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            is_add = not adj[u, v]
            
            dC5, dI5 = get_delta(u, v, is_add)
            dE = dC5 + dI5
            
            if dE < 0 or random.random() < np.exp(-dE / T):
                # Accept
                adj[u, v] = not adj[u, v]
                adj[v, u] = not adj[u, v]
                C5 += dC5
                I5 += dI5
                E += dE
                
                if E == 0:
                    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    break # Found valid graph for this restart
                    
            T *= cooling
            
    # If we didn't find a better graph, return seed
    return best_n, best_edges
