"""Model-and-Optimize approach for Heuristic ACO Orienteering."""
import numpy as np
import time
import random

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 320.0  # Safety margin before 330s kill switch
    
    N = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    d_j0 = distance_matrix[:, 0].copy()
    
    # Precompute: round-trip cost from i to j via 0 = d[i,j] + d[j,0]
    round_trip_cost = distance_matrix + d_j0[None, :]  # shape (N, N)
    
    def run_aco(eta, seed):
        """Run ACO with a given heuristic matrix and return (best_prize, all_prizes)."""
        np.random.seed(seed)
        
        tau = np.ones((N, N))
        all_prizes = []
        best_prize_global = 0.0
        
        for _ in range(30):
            iter_prizes = []
            best_prize_iter = 0.0
            
            for _ in range(20):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = np.zeros(N, dtype=bool)
                visited[0] = True
                
                while True:
                    remaining = max_len - path_len
                    costs = round_trip_cost[current, :]
                    mask = (~visited) & (costs <= remaining)
                    
                    if not np.any(mask):
                        break
                    
                    eligible_idx = np.where(mask)[0]
                    
                    weights = tau[current, eligible_idx] * eta[current, eligible_idx]
                    total_w = weights.sum()
                    
                    if total_w <= 0:
                        break
                    
                    probs = weights / total_w
                    r = np.random.rand()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible_idx):
                        idx = len(eligible_idx) - 1
                    
                    next_node = eligible_idx[idx]
                    path_len += distance_matrix[current, next_node]
                    prize_total += prizes[next_node]
                    visited[next_node] = True
                    current = next_node
                
                iter_prizes.append(prize_total)
                if prize_total > best_prize_iter:
                    best_prize_iter = prize_total
            
            if best_prize_iter > best_prize_global:
                best_prize_global = best_prize_iter
            all_prizes.extend(iter_prizes)
            
            tau = tau * 0.9
            tau += best_prize_iter / 10.0
        
        return best_prize_global, np.array(all_prizes)

    def build_heuristic(params):
        """Build heuristic matrix from parameters."""
        alpha, beta, gamma, delta, theta, blend = params
        eps = 1e-9
        
        P = np.maximum(prizes, eps)
        D = np.maximum(distance_matrix, eps)
        Dj0 = np.maximum(d_j0, eps)
        RT = np.maximum(round_trip_cost, eps)
        
        # Component 1: prize / distance^beta * return^gamma * RT^theta
        num1 = np.power(P[None, :], alpha)
        den1 = np.power(D, beta) * np.power(Dj0[None, :], gamma) * np.power(RT, theta)
        eta1 = num1 / den1
        eta1 = np.power(eta1, delta)
        
        # Component 2: pure prize (no distance penalty)
        eta2 = np.power(P[None, :], alpha * 0.5)
        
        # Blend
        eta = (1.0 - blend) * eta1 + blend * eta2
        eta = np.where(eta > 0, eta, 1e-9)
        eta = np.where(np.isfinite(eta), eta, 1e-9)
        
        return eta

    def soft_max_surrogate(all_prizes, temperature=0.1):
        """Smooth surrogate for max using log-sum-exp."""
        if len(all_prizes) == 0:
            return 0.0
        shifted = all_prizes - np.max(all_prizes)
        log_sum_exp = np.log(np.sum(np.exp(shifted / temperature))) * temperature
        return np.max(all_prizes) + log_sum_exp - np.log(len(all_prizes)) * temperature

    def evaluate_params(params, seeds):
        """Evaluate parameters across seeds, return mean soft-max score."""
        eta = build_heuristic(params)
        scores = []
        for s in seeds:
            _, all_prizes = run_aco(eta, s)
            score = soft_max_surrogate(all_prizes)
            scores.append(score)
        return np.mean(scores)

    # Parameter bounds: [alpha, beta, gamma, delta, theta, blend]
    bounds = [(0.1, 5.0), (0.1, 5.0), (0.0, 5.0), (0.1, 2.0), (0.0, 3.0), (0.0, 0.5)]
    
    def clip_params(p):
        p = list(p)
        for i, (lo, hi) in enumerate(bounds):
            p[i] = np.clip(p[i], lo, hi)
        return p
    
    # Phase 1: Coarse Grid Search
    grid_alpha = [0.5, 1.0, 2.0, 3.0]
    grid_beta = [0.5, 1.0, 2.0]
    grid_gamma = [0.0, 0.5, 1.0]
    grid_delta = [0.5, 1.0, 1.5]
    grid_theta = [0.0, 0.5, 1.0]
    grid_blend = [0.0, 0.25, 0.5]
    
    candidates = []
    for a in grid_alpha:
        for b in grid_beta:
            for g in grid_gamma:
                for d in grid_delta:
                    for t in grid_theta:
                        for bl in grid_blend:
                            candidates.append([a, b, g, d, t, bl])
    
    best_score_grid = -1
    best_params_grid = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    
    phase1_deadline = start_time + 55.0
    for p in candidates:
        if time.time() > phase1_deadline:
            break
        score = evaluate_params(p, [0])
        if score > best_score_grid:
            best_score_grid = score
            best_params_grid = p
    
    # Phase 2: Local Optimization using Nelder-Mead
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    if use_scipy:
        def objective(params):
            if time.time() > time_limit - 35:
                return 0.0
            return -evaluate_params(clip_params(params), [0, 1])
        
        x0 = np.array(best_params_grid, dtype=np.float64)
        
        res = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={'maxiter': 80, 'xatol': 0.02, 'fatol': 0.003, 'adaptive': True}
        )
        
        if time.time() < time_limit - 45:
            best_params_local = clip_params(res.x)
            best_score_local = -res.fun
        else:
            best_params_local = best_params_grid
            best_score_local = best_score_grid
    else:
        best_params_local = list(best_params_grid)
        best_score_local = best_score_grid
        
        for _ in range(60):
            if time.time() > time_limit - 45:
                break
            new_p = clip_params([x + random.uniform(-0.1, 0.1) for x in best_params_local])
            s = evaluate_params(new_p, [0])
            if s > best_score_local:
                best_params_local = new_p
                best_score_local = s

    # Phase 3: Final Verification with more seeds
    final_candidates = [
        (best_params_grid, "grid"),
        (best_params_local, "local"),
        ([1.0, 1.0, 1.0, 1.0, 0.0, 0.0], "init")
    ]
    
    best_final_params = best_params_local
    best_final_score = -np.inf
    
    phase3_deadline = time_limit - 25
    for p, name in final_candidates:
        if time.time() > phase3_deadline:
            break
        score = evaluate_params(clip_params(p), [0, 1, 2])
        if score > best_final_score:
            best_final_score = score
            best_final_params = clip_params(p)
    
    # Phase 4: High-precision polish
    if time.time() < time_limit - 40:
        polish_candidates = [best_final_params]
        for _ in range(8):
            if time.time() > time_limit - 40:
                break
            perturbed = clip_params([x + random.gauss(0, 0.03) for x in best_final_params])
            polish_candidates.append(perturbed)
        
        for p in polish_candidates:
            if time.time() > time_limit - 15:
                break
            score = evaluate_params(p, [0, 1, 2, 3, 4])
            if score > best_final_score:
                best_final_score = score
                best_final_params = p
    
    # Construct final heuristic
    eta = build_heuristic(best_final_params)
    
    # Safety check
    if not np.all(np.isfinite(eta)) or eta.shape != distance_matrix.shape:
        eta = (prizes[None, :] + 1e-9) / (distance_matrix + 1e-9)
        eta = np.where(eta > 0, eta, 1e-9)

    return eta
