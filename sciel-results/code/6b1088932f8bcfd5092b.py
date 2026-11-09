import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    best_n = 17
    best_edges = None
    # Seed: Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    seed_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_edges = seed_edges
    
    # Try to find larger graphs
    for n in range(35, 17, -1):
        if time.time() > deadline:
            break
        # Allocate time for this n
        # If n is large, give more time? Or equal?
        # Let's give 20s per n, decreasing for smaller n
        time_alloc = 20.0
        if n < 25:
            time_alloc = 10.0
        if n < 20:
            time_alloc = 5.0
            
        success, edges = _search_graph(n, time_alloc, start_time, deadline)
        if success:
            best_n = n
            best_edges = edges
            # If we found 35, we can stop? Or try to confirm?
            # 35 is the record. If we find it, it's likely the best possible.
            # But maybe 36? Unlikely.
            # Let's stop if we find 35 to save time for safety.
            if n >= 35:
                break
                
    return {'n': best_n, 'edges': best_edges}

def _search_graph(n, time_alloc, start_time, deadline):
    # Initialize random graph
    # Use a random regular-ish graph or just G(n, 0.5)
    # For K4-free, we want sparse. For no Ind6, we want dense.
    # R(4,6) ~ 36.
    # Let's start with p=0.5 and let search adjust.
    
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1
                
    current_penalty = _calc_penalty(adj, n)
    
    # Local Search
    # We want to minimize penalty.
    # Penalty = count_K4 + count_Ind6 (or boolean)
    # Let's use count_K4 + 10 * (1 if has_ind_6 else 0)
    
    no_improve_count = 0
    best_penalty_found = current_penalty
    best_adj = adj.copy()
    
    while time.time() < deadline:
        # Pick random edge
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        
        # Flip edge
        old_val = adj[i, j]
        adj[i, j] = 1 - old_val
        adj[j, i] = adj[i, j]
        
        new_penalty = _calc_penalty(adj, n)
        
        if new_penalty < current_penalty:
            current_penalty = new_penalty
            no_improve_count = 0
            if new_penalty < best_penalty_found:
                best_penalty_found = new_penalty
                best_adj = adj.copy()
                if best_penalty_found == 0:
                    # Convert to edges
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if best_adj[u, v]:
                                edges.append([u, v])
                    return True, edges
        else:
            # Revert
            adj[i, j] = old_val
            adj[j, i] = old_val
            no_improve_count += 1
            
            # Random restart if stuck
            if no_improve_count > 1000:
                # Reinitialize
                adj = np.zeros((n, n), dtype=np.uint8)
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < 0.5:
                            adj[u, v] = 1
                            adj[v, u] = 1
                current_penalty = _calc_penalty(adj, n)
                no_improve_count = 0
                
    return False, None

def _calc_penalty(adj, n):
    # Count K4
    k4_count = 0
    # Iterate all 4-tuples?
    # Optimize: Iterate pairs of edges?
    # Simple O(n^4) is fine for n=35 (1.5M tuples)
    # But Python loops are slow.
    # Use numpy?
    # Let's try a simple loop but break early if possible?
    # No, we need count for gradient.
    
    # Actually, for speed, let's just check existence for Ind6 and count K4.
    # If K4 count is high, penalty is high.
    
    # Vectorized K4 count?
    # A is adj matrix.
    # K4 exists if sum over all 4-tuples of prod(A[i,j]A[i,k]A[i,l]A[j,k]A[j,l]A[k,l]) > 0
    # This is hard to vectorize simply.
    
    # Let's use a faster heuristic:
    # 1. Check has_K4. If yes, return 1000 + count (approx).
    # 2. Check has_Ind6. If yes, return 100.
    
    # To make it smoother, let's count K4 properly.
    # For n=35, 35^4 is 1.5M.
    # In Python, 1.5M iterations is ~0.5s.
    # This is too slow for a tight loop (1000 steps/s).
    
    # We need a faster penalty.
    # How about: Penalty = (Number of edges in potential K4s)
    # Or just: 1 if has_K4 else 0.
    # And: 1 if has_Ind6 else 0.
    # Total penalty: 2 if both, 1 if one, 0 if none.
    # This is a step function. SA works poorly with step functions.
    
    # Alternative: Count the number of 4-cliques and 6-independent-sets.
    # Can we speed up the count?
    # For K4:
    # Iterate i < j < k < l.
    # Check A[i,j] & A[i,k] & A[i,l] & A[j,k] & A[j,l] & A[k,l].
    
    # Let's try to optimize the Python loop.
    # Or use a different metric.
    # Metric: Sum over all triangles of (1 - min_degree_of_triangle)? No.
    
    # Let's stick to boolean for now but use a "kick" strategy.
    # If we can't find a valid graph, we return False.
    
    # To speed up, I will implement a simple check that returns 0, 1, or 2.
    # And rely on random restarts.
    
    has_k4 = _has_k4(adj, n)
    has_ind6 = _has_ind6(adj, n)
    
    if has_k4 and has_ind6:
        return 2
    elif has_k4 or has_ind6:
        return 1
    else:
        return 0

def _has_k4(adj, n):
    # Check for any K4
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i, j]:
                continue
            for k in range(j + 1, n):
                if not adj[i, k] or not adj[j, k]:
                    continue
                for l in range(k + 1, n):
                    if adj[i, l] and adj[j, l] and adj[k, l]:
                        return True
    return False

def _has_ind6(adj, n):
    # Check if complement has K6
    # Complement edge exists if adj[i,j] == 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                continue
            for k in range(j + 1, n):
                if adj[i, k] or adj[j, k]:
                    continue
                for l in range(k + 1, n):
                    if adj[i, l] or adj[j, l] or adj[k, l]:
                        continue
                    for m in range(l + 1, n):
                        if adj[i, m] or adj[j, m] or adj[k, m] or adj[l, m]:
                            continue
                        for p in range(m + 1, n):
                            if adj[i, p] or adj[j, p] or adj[k, p] or adj[l, p] or adj[m, p]:
                                continue
                            return True
    return False
