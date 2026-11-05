def construct():
    import random
    import time

    n = 18
    
    # Optimized max triangle-free induced subgraph finder
    def max_triangle_free(adj_masks):
        best = 0
        # Order vertices by degree (descending) to prune earlier?
        # Or just use standard order.
        # Let's stick to standard order but use aggressive pruning.
        
        def has_edge_in_mask(mask):
            m = mask
            while m:
                v_bit = m & -m
                v = v_bit.bit_length() - 1
                if adj_masks[v] & (mask ^ v_bit):
                    return True
                m ^= v_bit
            return False

        def backtrack(idx, mask, count):
            nonlocal best
            if count > best:
                best = count
            if idx == n:
                return
            
            # Prune: even if we take all remaining, we can't beat best
            if count + (n - idx) <= best:
                return
            
            # Option 1: Skip vertex idx
            backtrack(idx + 1, mask, count)
            
            # Option 2: Include vertex idx
            N = mask & adj_masks[idx]
            if not has_edge_in_mask(N):
                backtrack(idx + 1, mask | (1 << idx), count + 1)

        backtrack(0, 0, 0)
        return best

    def is_k4_free(adj_masks):
        for i in range(n):
            for j in range(i+1, n):
                if adj_masks[i] & (1 << j):
                    # Check if common neighbors have an edge
                    common = adj_masks[i] & adj_masks[j]
                    if has_edge_in_mask(common):
                        return False
        return True

    def generate_initial(p, seed):
        random.seed(seed)
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Prune K4s
        while True:
            k4_found = False
            for a in range(n):
                if k4_found: break
                for b in range(a+1, n):
                    if k4_found: break
                    if not adj[a][b]: continue
                    for c in range(b+1, n):
                        if k4_found: break
                        if not adj[a][c] or not adj[b][c]: continue
                        for d in range(c+1, n):
                            if adj[a][d] and adj[b][d] and adj[c][d]:
                                k4_found = True
                                u, v = random.choice([(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)])
                                adj[u][v] = False
                                adj[v][u] = False
                                break
            if not k4_found:
                break
        
        adj_masks = [0]*n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j]:
                    mask |= (1 << j)
            adj_masks[i] = mask
        return adj_masks

    def try_improve(adj_masks, time_limit=50):
        # Greedy edge addition
        # We want to add edges if it doesn't create K4 and reduces score
        current_score = max_triangle_free(adj_masks)
        improved = True
        start_time = time.time()
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj_masks[u] & (1 << v):
                        continue # Edge exists
                    
                    # Check if adding (u,v) creates K4
                    # K4 if common neighbors of u and v contain an edge
                    common = adj_masks[u] & adj_masks[v]
                    if has_edge_in_mask(common):
                        continue # Would create K4
                    
                    # Try adding edge
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    
                    new_score = max_triangle_free(adj_masks)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        adj_masks[u] ^= (1 << v)
                        adj_masks[v] ^= (1 << u)
                        
        return adj_masks, current_score

    best_edges = None
    best_score = 999

    # Try multiple initial configurations
    seeds = [42, 123, 456, 789, 1000]
    ps = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for seed in seeds:
        for p in ps:
            adj_masks = generate_initial(p, seed)
            adj_masks, score = try_improve(adj_masks, time_limit=10)
            if score < best_score:
                best_score = score
                # Convert back to edges
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj_masks[i] & (1 << j):
                            edges.append([i, j])
                best_edges = edges
                
            if best_score <= 10: # Can't get much lower, likely 10 or 11
                break
        if best_score <= 10:
            break

    # Fallback to Turan if nothing found (shouldn't happen)
    if best_edges is None:
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        t_edges = []
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        t_edges.append([a, b])
        best_edges = t_edges

    return best_edges
