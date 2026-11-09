import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Try to find a graph on n=39
    # If successful, return it.
    # Else, fall back to n=26 (seed)
    
    def check_validity(n, edges):
        # Check triangle-free
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Triangle check
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    # Check common neighbors
                    common = adj[u] & adj[v]
                    # Mask out u and v
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common:
                        return False, False
        
        # Check independent set of size 10
        # Use branch and bound
        def has_ind_set_10():
            # Simple backtracking
            # State: available vertices, current set size
            # Optimized with bitsets
            
            # We want to know if there is an independent set of size 10.
            # This is equivalent to a clique of size 10 in the complement.
            # Complement adjacency:
            comp_adj = [0] * n
            for i in range(n):
                # comp_adj[i] has bits for j not connected to i in G
                # All bits set, then clear adj[i] and i
                comp_adj[i] = ((1 << n) - 1) ^ adj[i] ^ (1 << i)
            
            # Find clique of size 10 in comp_adj
            # Use a simple recursive search with pruning
            
            def dfs(candidates, depth):
                if depth == 10:
                    return True
                if not candidates:
                    return False
                # Heuristic: pick vertex with highest degree in candidates
                # Or just iterate
                # To speed up, we can use a greedy upper bound on clique size
                # But for n=39, simple B&B might be slow if not careful.
                
                # A faster check:
                # If the maximum degree in the subgraph induced by candidates is low, prune.
                
                # Let's use a standard B&B for clique
                # candidates is a bitset
                while candidates:
                    # Pick a vertex
                    # To improve, pick vertex with max degree in candidates
                    v = -1
                    max_deg = -1
                    temp = candidates
                    while temp:
                        # Get lowest set bit
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        deg = (comp_adj[idx] & candidates).bit_count()
                        if deg > max_deg:
                            max_deg = deg
                            v = idx
                        temp ^= lsb
                    
                    if v == -1:
                        break
                    
                    # Branch 1: Include v
                    # New candidates are neighbors of v in comp_adj that are in candidates
                    new_cands = candidates & comp_adj[v]
                    if dfs(new_cands, depth + 1):
                        return True
                    
                    # Branch 2: Exclude v
                    candidates &= ~(1 << v)
                    
                    # Pruning: if remaining candidates + depth < 10, fail
                    if candidates.bit_count() + depth < 10:
                        return False
                        
                return False
            
            return dfs((1 << n) - 1, 0)

        is_valid = not has_ind_set_10()
        return True, is_valid

    def try_n(n):
        # Local search
        for _ in range(50): # 50 restarts
            if time.time() > deadline:
                break
                
            # Initialize random graph
            edges = []
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        edges.append([i, j])
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Remove triangles
            changed = True
            while changed:
                changed = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            common = adj[u] & adj[v]
                            common &= ~(1 << u)
                            common &= ~(1 << v)
                            if common:
                                # Remove edge (u,v)
                                # Find a w in common
                                w = (common & -common).bit_length() - 1
                                # Remove (u,v)
                                edges.remove([u, v])
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                changed = True
                                break
                    if changed:
                        break
            
            # Add edges greedily
            for _ in range(100): # 100 steps of adding
                if time.time() > deadline:
                    break
                # Pick random non-edge
                for _ in range(20):
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v: continue
                    if u > v: u, v = v, u
                    if not ((adj[u] >> v) & 1):
                        # Check if adding creates triangle
                        common = adj[u] & adj[v]
                        common &= ~(1 << u)
                        common &= ~(1 << v)
                        if not common:
                            # Add edge
                            edges.append([u, v])
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            break
            
            # Check validity
            valid, alpha_ok = check_validity(n, edges)
            if valid and alpha_ok:
                return {'n': n, 'edges': edges}
        
        return None

    # Try n=39
    result = try_n(39)
    if result:
        return result
    
    # Fallback to n=26
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
