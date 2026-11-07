import numpy as np
import time

def _run_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=0):
    """
    Runs the ACO simulation to evaluate the heuristic.
    """
    n = len(prizes)
    rng = np.random.default_rng(seed)
    
    eta = heuristic.copy()
    # Ensure positive values as per task description
    eta[eta <= 0] = 1e-9
    
    tau = np.ones((n, n))
    
    d = distance_matrix
    d_j0 = d[0, :] 
    # Precompute cost matrix for validity check: d[i,j] + d[j,0]
    cost = d + d_j0[None, :]
    
    best_global = 0.0
    
    for _ in range(num_iters):
        best_iter = 0.0
        
        for _ in range(num_ants):
            cur = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            length = 0.0
            prize = 0.0
            
            while True:
                remaining = max_len - length
                # Valid nodes j: not visited, and length + d[cur, j] + d[j, 0] <= max_len
                valid_mask = (~visited) & (cost[cur, :] <= remaining)
                valid_nodes = np.where(valid_mask)[0]
                
                if len(valid_nodes) == 0:
                    break
                
                w = tau[cur, valid_nodes] * eta[cur, valid_nodes]
                sum_w = w.sum()
                if sum_w < 1e-10:
                    break
                p = w / sum_w
                
                next_node = rng.choice(valid_nodes, p=p)
                
                visited[next_node] = True
                length += d[cur, next_node]
                prize += prizes[next_node]
                cur = next_node
            
            if prize > best_iter:
                best_iter = prize
        
        if best_iter > best_global:
            best_global = best_iter
            
        # Pheromone update
        tau = 0.9 * tau + (best_iter / 10.0)
        
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    # Hard deadline of 280s to be safe within 300s limit
    deadline = start_time + 280.0
    
    n = len(prizes)
    d = distance_matrix
    p = prizes
    eps = 1e-9
    
    d_j0 = d[0, :]
    
    # Strategy: Opportunity Cost / Regret Heuristic
    # Attractiveness H[i, j] is high if p[j] is high AND there are few "better" nodes k 
    # that are "cheaper" to visit from i.
    # "Cheaper" is defined by total cost C[i, k] = d[i, k] + d[k, 0].
    # If a node k has C[i, k] < C[i, j] and p[k] > p[j], then visiting j instead of k 
    # incurs an opportunity cost. We penalize j based on the maximum prize of such k.
    
    # Precompute total cost matrix C
    C = d + d_j0[None, :]
    
    # We want H[i, j] = p[j] / (1 + MaxPrizeBetterCheaper(i, j))^alpha
    # MaxPrizeBetterCheaper(i, j) = max{ p[k] | C[i, k] < C[i, j] and p[k] > p[j] }
    
    # To compute this efficiently:
    # For each i, sort nodes by C[i, :].
    # For a given j, we look at all k with C[i, k] < C[i, j].
    # Among those, we want the max p[k] that is still > p[j].
    # Actually, if max p[k] among cheaper nodes is M, and M > p[j], then penalty is M.
    # If M <= p[j], penalty is 0 (no better cheaper node).
    # So Penalty(i, j) = max(0, max{p[k] | C[i, k] < C[i, j]} - p[j])? 
    # No, the definition was "max prize of any other unvisited node k that is also reachable... and closer".
    # If the max prize among cheaper nodes is M, and M > p[j], then M is the opportunity cost.
    # If M <= p[j], then j is the best among cheaper nodes, so opportunity cost is 0? 
    # Or is it relative to p[j]?
    # Let's use: Penalty = max(0, MaxPrizeCheaper - p[j]).
    # Then H[i, j] = p[j] / (1 + Penalty)^beta.
    
    # This can be computed in O(N^2 log N) per instance or O(N^3) simply.
    # N=50, N^3 = 125k ops. Very fast.
    
    H = np.zeros((n, n))
    
    for i in range(n):
        # Get costs from i to all j
        costs_i = C[i, :]
        
        # For each j, find max prize of k such that costs_i[k] < costs_i[j]
        # We can sort indices by cost.
        sorted_idx = np.argsort(costs_i)
        sorted_costs = costs_i[sorted_idx]
        sorted_prizes = p[sorted_idx]
        
        # For each position in sorted order (which corresponds to a node j),
        # the "cheaper" nodes are those before it in the sorted list.
        # We need the max prize among those.
        # Let max_prize_so_far[j_index_in_sorted] = max(sorted_prizes[:j_index_in_sorted])
        
        # We can compute running max.
        # Note: strictly less than. If costs are equal, they are not "cheaper".
        # With floating point, exact equality is rare, but let's be safe.
        # If we use <, then for the first element, no cheaper nodes.
        
        max_prize_cheaper = np.zeros(n)
        current_max = 0.0
        
        # We iterate through the sorted nodes.
        # For the node at sorted_idx[t], its cost is sorted_costs[t].
        # The nodes cheaper than it are sorted_idx[0:t] (assuming strict inequality).
        # If there are ties in cost, this simple loop might include ties.
        # To handle strict inequality properly:
        
        # Let's just do the O(N^2) check for each j to be robust and simple.
        # For each j in 0..n-1:
        #   mask = costs_i < costs_i[j]
        #   if any mask:
        #       max_p = max(p[mask])
        #       penalty = max(0, max_p - p[j])
        #   else:
        #       penalty = 0
        #   H[i, j] = p[j] / (1 + penalty)
        
        for j in range(n):
            if j == i:
                continue # Self loop, though usually not picked if visited, but good to be clean
                
            mask_cheaper = costs_i < costs_i[j]
            if np.any(mask_cheaper):
                max_p_cheaper = np.max(p[mask_cheaper])
                penalty = max(0.0, max_p_cheaper - p[j])
            else:
                penalty = 0.0
            
            # Heuristic formula
            # Base attractiveness is prize.
            # Penalized by opportunity cost.
            # Also add a small distance penalty to break ties or guide slightly?
            # The regret metric already accounts for "closeness" via cost.
            # Let's stick to pure regret/opportunity cost.
            
            H[i, j] = p[j] / (1.0 + penalty)
            
    # Ensure positive values
    H[H <= 0] = eps
    
    # Fallback if something went wrong
    if not np.all(np.isfinite(H)) or H.shape != (n, n):
        H = p[None, :] / (C + eps)
        H[H <= 0] = eps
        return H

    # We can also try to blend this with a standard heuristic to ensure stability
    # if the pure regret heuristic is too aggressive or sparse.
    # H_std = p[None, :] / (C + eps)
    # H_std[H_std <= 0] = eps
    
    # Let's evaluate a few blends to see if one performs better in the proxy ACO.
    # Candidates:
    # 1. Pure Regret
    # 2. Regret * Standard
    # 3. 0.5 Regret + 0.5 Standard
    
    candidates = [H.copy()]
    
    H_std = p[None, :] / (C + eps)
    H_std[H_std <= 0] = eps
    
    # Normalize for blending?
    # H and H_std have different scales.
    # Let's just try multiplicative and additive blends on normalized scales.
    
    H_norm = H / (H.max() + eps)
    H_std_norm = H_std / (H_std.max() + eps)
    
    candidates.append(H_norm * H_std_norm) # Multiplicative blend (both must be high)
    candidates.append(0.5 * H_norm + 0.5 * H_std_norm) # Additive blend
    
    # Evaluate candidates with a quick ACO run
    best_score = -1
    best_heuristic = candidates[0]
    
    # Time check
    if time.time() > deadline - 30:
        return best_heuristic
    
    # Use a reduced ACO for selection to save time
    # 10 ants, 10 iters should be enough to distinguish good heuristics
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        
        cand_clean = cand.copy()
        cand_clean[cand_clean <= 0] = eps
        
        try:
            score = _run_aco(p, d, max_len, cand_clean, num_ants=10, num_iters=10, seed=0)
            if score > best_score:
                best_score = score
                best_heuristic = cand_clean
        except Exception:
            continue
            
    return best_heuristic
