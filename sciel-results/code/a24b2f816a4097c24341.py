import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        # Work on a copy of adjacency for remaining edges
        rem_adj = adj[:] 
        cliques = []
        
        # Helper to get list of set bits
        def bits(mask):
            res = []
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                res.append(idx)
                mask ^= lsb
            return res

        # Main loop
        while True:
            # Check if any edges remain
            has_edges = False
            for u in range(n):
                if rem_adj[u]:
                    has_edges = True
                    break
            if not has_edges:
                break
                
            # Find best seed edge
            # Heuristic: Edge with max common neighbors
            best_edge = None
            max_common = -1
            # Iterate edges
            for u in range(n):
                mask_u = rem_adj[u]
                # Only look at v > u to avoid duplicates
                mask_u &= ~((1 << (u + 1)) - 1) # mask out <= u
                while mask_u:
                    lsb = mask_u & -mask_u
                    v = lsb.bit_length() - 1
                    common = rem_adj[u] & rem_adj[v]
                    common_count = common.bit_count()
                    if common_count > max_common:
                        max_common = common_count
                        best_edge = (u, v)
                    mask_u ^= lsb
            
            if best_edge is None:
                break # Should not happen if has_edges was True, but safety
                
            u, v = best_edge
            # Build maximal clique starting with {u, v}
            clique = [u, v]
            candidates = rem_adj[u] & rem_adj[v]
            
            # Greedy expansion
            while candidates:
                # Pick candidate with max degree within candidates
                best_w = -1
                max_deg = -1
                temp_cand = candidates
                while temp_cand:
                    lsb = temp_cand & -temp_cand
                    w = lsb.bit_length() - 1
                    # Degree of w within candidates
                    deg = (rem_adj[w] & candidates).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        best_w = w
                    temp_cand ^= lsb
                
                if best_w == -1: break
                clique.append(best_w)
                # Update candidates
                candidates &= rem_adj[best_w]
                
            # Record clique
            cliques.append(clique)
            
            # Remove edges of this clique from rem_adj
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    rem_adj[a] &= ~(1 << b)
                    rem_adj[b] &= ~(1 << a)
                    
            # Time check
            if time.time() > deadline:
                break
                
        return cliques

    results = []
    for params in [(32, 0.6, 601), (64, 0.55, 602)]:
        if time.time() > deadline:
            # If we run out of time, we must return *something* valid
            # But the greedy is fast. If it fails, return empty or seed-like?
            # Returning invalid is score 0.
            # We assume it finishes.
            pass
        results.append(solve_instance(*params))
        
    return results
