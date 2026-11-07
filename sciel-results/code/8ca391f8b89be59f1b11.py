import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 320.0  # Safety margin
    
    N = len(prizes)
    d_j0 = distance_matrix[:, 0].copy()
    
    # Precompute eligibility masks for each possible current node and remaining length?
    # No, remaining length changes.
    # But we can precompute d[i, j] + d[j, 0] for all i, j.
    # Let cost_ij = d[i, j] + d[j, 0]
    # Condition: path_len + cost_ij <= max_len
    cost_to_return = distance_matrix + d_j0[None, :]
    
    def run_aco(eta, seed):
        np.random.seed(seed)
        tau = np.ones((N, N))
        
        best_prize_global = 0.0
        
        # Precompute log(eta) for speed? No, eta changes? No, eta is static per instance.
        # But tau changes.
        # log(tau * eta) = log(tau) + log(eta)
        log_eta = np.log(eta)
        
        for it in range(30):
            best_prize_iter = 0.0
            
            for ant in range(20):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = [0]
                # Using a set for visited is slow. Use a boolean array?
                # N=50, so a list check `if j in visited` is O(N).
                # A boolean array `is_visited` is O(1) check.
                is_visited = np.zeros(N, dtype=bool)
                is_visited[0] = True
                
                while True:
                    # Find eligible nodes
                    # eligible if not visited AND path_len + cost[current, j] <= max_len
                    # cost[current, :] is a vector.
                    eligible_mask = (~is_visited) & (path_len + cost_to_return[current, :] <= max_len)
                    eligible_idx = np.where(eligible_mask)[0]
                    
                    if len(eligible_idx) == 0:
                        break
                    
                    # Calculate probabilities
                    # p_j = tau[current, j] * eta[current, j]
                    # We need to normalize over eligible_idx
                    weights = tau[current, eligible_idx] * eta[current, eligible_idx]
                    
                    # Handle zero weights (shouldn't happen if eta > 0 and tau > 0)
                    total_w = weights.sum()
                    if total_w <= 0:
                        break
                    
                    probs = weights / total_w
                    
                    # Sample
                    r = np.random.rand()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible_idx):
                        idx = len(eligible_idx) - 1
                    
                    next_node = eligible_idx[idx]
                    path_len += distance_matrix[current, next_node]
                    prize_total += prizes[next_node]
                    is_visited[next_node] = True
                    current = next_node
                
                if prize_total > best_prize_iter:
                    best_prize_iter = prize_total
            
            if best_prize_iter > best_prize_global:
                best_prize_global = best_prize_iter
            
            tau = tau * 0.9
            tau += best_prize_iter / 10.0
        
        return best_prize_global

    def gen_heuristic(gamma, beta, delta, eps=1e-6):
        # eta_ij = p_j^gamma / (d_ij^beta * d_j0^delta)
        # Note: d_ij is distance_matrix[i, j]
        # d_j0 is distance_matrix[j, 0]
        
        p_pow = np.power(np.maximum(prizes, eps), gamma)[None, :]
        d_pow = np.power(np.maximum(distance_matrix, eps), beta)
        d0_pow = np.power(np.maximum(d_j0, eps), delta)[None, :]
        
        denom = d_pow * d0_pow
        eta = p_pow / denom
        
        # Ensure positive
        eta = np.where(eta > 0, eta, 1e-9)
        return eta

    # Grid search
    # gamma: 0.5, 1.0, 1.5, 2.0
    # beta: 0.5, 1.0, 1.5
    # delta: 0.5, 1.0, 1.5, 2.0
    
    params = []
    for g in [0.5, 1.0, 1.5, 2.0]:
        for b in [0.5, 1.0, 1.5]:
            for d in [0.5, 1.0, 1.5, 2.0]:
                params.append((g, b, d))
                
    best_score = -1
    best_params = None
    best_eta = None
    
    # Phase 1: Quick screen with seed 0
    quick_results = []
    for g, b, d in params:
        if time.time() - start_time > time_limit - 60:
            break
        eta = gen_heuristic(g, b, d)
        s = run_aco(eta, 0)
        quick_results.append((s, g, b, d, eta))
        
    quick_results.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Evaluate top candidates with seeds 1, 2
    top_k = min(10, len(quick_results))
    for s_quick, g, b, d, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 20:
            break
        scores = [s_quick]
        for seed in [1, 2]:
            if time.time() - start_time > time_limit - 10:
                break
            s = run_aco(eta, seed)
            scores.append(s)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = (g, b, d)
            best_eta = eta
            
    # Phase 3: Fine-tune around best params
    if best_params is not None and time.time() - start_time < time_limit - 30:
        g0, b0, d0 = best_params
        fine_params = []
        for dg in [-0.2, -0.1, 0.1, 0.2]:
            for db in [-0.2, -0.1, 0.1, 0.2]:
                for dd in [-0.2, -0.1, 0.1, 0.2]:
                    ng = g0 + dg
                    nb = b0 + db
                    nd = d0 + dd
                    if ng > 0 and nb > 0 and nd > 0:
                        fine_params.append((ng, nb, nd))
                        
        for g, b, d in fine_params:
            if time.time() - start_time > time_limit - 10:
                break
            eta = gen_heuristic(g, b, d)
            # Quick check with seed 0
            s0 = run_aco(eta, 0)
            # Only if it's promising (close to best or better)
            if s0 >= best_score * 0.95: 
                scores = [s0]
                for seed in [1, 2]:
                    if time.time() - start_time > time_limit - 5:
                        break
                    s = run_aco(eta, seed)
                    scores.append(s)
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = (g, b, d)
                    best_eta = eta

    if best_eta is None:
        # Fallback
        best_eta = gen_heuristic(1.0, 1.0, 1.0)

    return best_eta
