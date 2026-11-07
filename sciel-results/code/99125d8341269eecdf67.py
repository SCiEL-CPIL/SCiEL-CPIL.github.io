"""Model-and-Optimize approach for Heuristic ACO Orienteering."""
import numpy as np
import time
import random

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 325.0  # Safety margin before 330s kill switch
    
    N = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    d_j0 = distance_matrix[:, 0].copy()
    
    def evaluate_heuristic(params, seed):
        """
        Run ACO with a specific heuristic parameterization and return best prize.
        params: [alpha, beta, gamma, delta]
        Heuristic: eta_ij = ( P_j^alpha / ( (D_ij + eps)^beta * (D_j0 + eps)^gamma ) )^delta
        """
        np.random.seed(seed)
        
        alpha, beta, gamma, delta = params
        
        # Compute Heuristic Matrix
        eps = 1e-9
        
        P = np.maximum(prizes, eps)
        D = np.maximum(distance_matrix, eps)
        Dj0 = np.maximum(d_j0, eps)
        
        num = np.power(P[None, :], alpha)
        den = np.power(D, beta) * np.power(Dj0[None, :], gamma)
        
        eta_base = num / den
        eta = np.power(eta_base, delta)
        
        # Enforce positivity and replace non-positive with 1e-9
        eta = np.where(eta > 0, eta, 1e-9)
        
        # ACO Simulation
        tau = np.ones((N, N))
        best_prize_global = 0.0
        
        for _ in range(30):
            best_prize_iter = 0.0
            
            for _ in range(20):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = np.zeros(N, dtype=bool)
                visited[0] = True
                
                while True:
                    # Find eligible nodes
                    costs = distance_matrix[current, :] + d_j0
                    remaining = max_len - path_len
                    mask = (~visited) & (costs <= remaining)
                    
                    if not np.any(mask):
                        break
                    
                    eligible_idx = np.where(mask)[0]
                    
                    # Weights: tau[i, j] * eta[i, j]
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
                
                if prize_total > best_prize_iter:
                    best_prize_iter = prize_total
            
            if best_prize_iter > best_prize_global:
                best_prize_global = best_prize_iter
            
            tau = tau * 0.9
            tau += best_prize_iter / 10.0
        
        return best_prize_global

    def multi_seed_eval(params, seeds):
        """Evaluate params across multiple seeds to reduce variance."""
        scores = []
        for s in seeds:
            score = evaluate_heuristic(params, s)
            scores.append(score)
        return np.mean(scores)

    # Define parameter bounds and initial guesses
    initial_x0 = [1.0, 1.0, 1.0, 1.0]
    
    # Phase 1: Grid Search to find a good basin
    # Sample points in the parameter space
    grid_alpha = [0.5, 1.0, 1.5, 2.0, 3.0]
    grid_beta = [0.5, 1.0, 1.5, 2.0, 3.0]
    grid_gamma = [0.0, 0.5, 1.0, 1.5]
    grid_delta = [0.5, 1.0, 1.5]
    
    candidates = []
    for a in grid_alpha:
        for b in grid_beta:
            for g in grid_gamma:
                for d in grid_delta:
                    candidates.append([a, b, g, d])
    
    # Evaluate with seed 0 only for speed in phase 1
    best_score_grid = -1
    best_params_grid = initial_x0
    
    # Limit grid search size if it's too big (5*5*4*3 = 300 points)
    for p in candidates:
        if time.time() - start_time > time_limit - 60:
            break
        score = evaluate_heuristic(p, 0)
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
        # Objective function to minimize (negative of mean score)
        def objective(params):
            if time.time() - start_time > time_limit - 10:
                return 0.0
            # Use 2 seeds for stability during optimization
            return -multi_seed_eval(params, [0, 1])
        
        # Start from best grid point
        x0 = best_params_grid
        
        # Bounds:
        bounds = [(0.1, 5.0), (0.1, 5.0), (0.0, 5.0), (0.1, 2.0)]
        
        res = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={'maxiter': 50, 'xatol': 0.05, 'fatol': 0.01}
        )
        
        if time.time() - start_time < time_limit - 20:
            best_params_local = res.x
            best_score_local = -res.fun
        else:
            best_params_local = x0
            best_score_local = -objective(x0)
            
    else:
        # Fallback: simple hill climbing
        best_params_local = best_params_grid
        best_score_local = best_score_grid
        
        for _ in range(50):
            if time.time() - start_time > time_limit - 20:
                break
            new_p = [x + random.uniform(-0.2, 0.2) for x in best_params_local]
            new_p[0] = np.clip(new_p[0], 0.1, 5.0)
            new_p[1] = np.clip(new_p[1], 0.1, 5.0)
            new_p[2] = np.clip(new_p[2], 0.0, 5.0)
            new_p[3] = np.clip(new_p[3], 0.1, 2.0)
            
            s = evaluate_heuristic(new_p, 0)
            if s > best_score_local:
                best_params_local = new_p
                best_score_local = s

    # Phase 3: Final Verification and Selection
    final_candidates = [
        (best_params_grid, "grid"),
        (best_params_local, "local")
    ]
    
    # Add the initial guess just in case
    final_candidates.append((initial_x0, "init"))
    
    best_final_params = best_params_local
    best_final_score = -np.inf
    
    for p, name in final_candidates:
        if time.time() - start_time > time_limit - 5:
            break
        # Evaluate with seeds 0, 1, 2
        score = multi_seed_eval(p, [0, 1, 2])
        if score > best_final_score:
            best_final_score = score
            best_final_params = p
    
    # Construct final heuristic
    alpha, beta, gamma, delta = best_final_params
    eps = 1e-9
    
    P = np.maximum(prizes, eps)
    D = np.maximum(distance_matrix, eps)
    Dj0 = np.maximum(d_j0, eps)
    
    num = np.power(P[None, :], alpha)
    den = np.power(D, beta) * np.power(Dj0[None, :], gamma)
    
    eta_base = num / den
    eta = np.power(eta_base, delta)
    
    eta = np.where(eta > 0, eta, 1e-9)
    
    # Safety check
    if not np.all(np.isfinite(eta)) or eta.shape != distance_matrix.shape:
        # Fallback to simple prize/distance
        eta = (prizes[None, :] + eps) / (distance_matrix + eps)
        eta = np.where(eta > 0, eta, 1e-9)

    return eta
