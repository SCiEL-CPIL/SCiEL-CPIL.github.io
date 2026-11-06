import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    def instance(n, p, seed):
        # Use random module as in seed
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def min_vertex_cover(edges, n):
        m = len(edges)
        if m == 0:
            return []
            
        # Precompute incidence masks
        inc_mask = [0] * n
        for i, (u, v) in enumerate(edges):
            bit = 1 << i
            inc_mask[u] |= bit
            inc_mask[v] |= bit
            
        all_edges_mask = (1 << m) - 1
        
        best_size = m
        best_cover = []
        
        # Greedy Upper Bound
        # Pick vertex with max degree
        current_E = all_edges_mask
        greedy_cover = []
        while current_E:
            # Find max degree vertex
            max_deg = -1
            max_v = -1
            for v in range(n):
                deg = (current_E & inc_mask[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    max_v = v
            if max_v == -1: break
            greedy_cover.append(max_v)
            current_E &= ~inc_mask[max_v]
            
        best_size = len(greedy_cover)
        best_cover = greedy_cover[:]
        
        # B&B
        # To avoid recursion depth issues, we could use iterative, but n=40 is small.
        # Recursion depth is at most best_size.
        
        def solve(E_mask, size, cover):
            nonlocal best_size, best_cover, deadline
            
            if time.time() > deadline:
                return
                
            if E_mask == 0:
                if size < best_size:
                    best_size = size
                    best_cover = cover[:]
                return
                
            # Lower Bound: Greedy Maximal Matching
            rem_E = E_mask
            match_count = 0
            while rem_E:
                lsb = rem_E & -rem_E
                idx = lsb.bit_length() - 1
                u_m, v_m = edges[idx]
                match_count += 1
                rem_E &= ~(inc_mask[u_m] | inc_mask[v_m])
                
            if size + match_count >= best_size:
                return
                
            # Branching Heuristic
            # Find edge with max deg(u) + deg(v)
            max_score = -1
            branch_idx = -1
            # Optimization: If we have many edges, iterating all is ok.
            for i in range(m):
                if (E_mask >> i) & 1:
                    u, v = edges[i]
                    deg_u = (E_mask & inc_mask[u]).bit_count()
                    deg_v = (E_mask & inc_mask[v]).bit_count()
                    score = deg_u + deg_v
                    if score > max_score:
                        max_score = score
                        branch_idx = i
                        
            if branch_idx == -1:
                return
                
            u, v = edges[branch_idx]
            
            # Branch 1: Include u
            # Heuristic: Try the one with higher degree first?
            deg_u = (E_mask & inc_mask[u]).bit_count()
            deg_v = (E_mask & inc_mask[v]).bit_count()
            
            if deg_u >= deg_v:
                # Branch on u first
                new_E_1 = E_mask & ~inc_mask[u]
                solve(new_E_1, size + 1, cover + [u])
                new_E_2 = E_mask & ~inc_mask[v]
                solve(new_E_2, size + 1, cover + [v])
            else:
                # Branch on v first
                new_E_2 = E_mask & ~inc_mask[v]
                solve(new_E_2, size + 1, cover + [v])
                new_E_1 = E_mask & ~inc_mask[u]
                solve(new_E_1, size + 1, cover + [u])
                
        solve(all_edges_mask, 0, [])
        return sorted(best_cover)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        cover = min_vertex_cover(edges, n)
        out.append(cover)
        
    return out
