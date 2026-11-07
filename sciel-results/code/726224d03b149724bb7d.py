# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 290.0  # Safety margin
    
    N = len(prizes)
    d_j0 = distance_matrix[:, 0]  # distance from node 0 to each node (symmetric)
    # d_j0[j] = distance_matrix[0, j] = distance_matrix[j, 0]
    
    # Precompute prize vector (node 0 has prize 0)
    prizes_arr = np.asarray(prizes, dtype=np.float64)
    dm = np.asarray(distance_matrix, dtype=np.float64)
    
    # Run a single ACO instance with given heuristic matrix and seed
    def run_aco(eta, seed):
        np.random.seed(seed)
        tau = np.ones((N, N))
        best_prize_global = 0.0
        
        for it in range(30):
            best_prize_iter = 0.0
            
            for ant in range(20):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = [0]
                
                while True:
                    eligible = []
                    for j in range(N):
                        if j in visited:
                            continue
                        if path_len + dm[current, j] + dm[j, 0] <= max_len:
                            eligible.append(j)
                    
                    if not eligible:
                        break
                    
                    total_p = 0.0
                    probs = np.zeros(len(eligible))
                    for k, j in enumerate(eligible):
                        p = tau[current, j] * eta[current, j]
                        probs[k] = p
                        total_p += p
                    
                    if total_p <= 0:
                        break
                    
                    probs /= total_p
                    r = np.random.rand()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible):
                        idx = len(eligible) - 1
                    
                    next_node = eligible[idx]
                    path_len += dm[current, next_node]
                    prize_total += prizes_arr[next_node]
                    visited.append(next_node)
                    current = next_node
            
            if prize_total > best_prize_iter:
                best_prize_iter = prize_total
            
            if best_prize_iter > best_prize_global:
                best_prize_global = best_prize_iter
            
            tau = tau * 0.9
            tau += best_prize_iter / 10.0
        
        return best_prize_global
    
    # Heuristic generation functions
    def make_eta(params):
        """
        params: dict with keys:
          - alpha, beta: exponents for prize and distance
          - gamma: weight on return distance
          - form: 0 = prize^alpha / (dist * (1 + gamma*d_j0))^beta
                  1 = prize^alpha / (dist + gamma*d_j0)^beta
        """
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 1.0)
        gamma = params.get('gamma', 1.0)
        form = params.get('form', 0)
        
        # Avoid division by zero
        dm_safe = np.maximum(dm, 1e-9)
        prizes_safe = np.maximum(prizes_arr[None, :], 1e-9)
        
        if form == 0:
            denom = dm_safe * (1.0 + gamma * d_j0[None, :])
        else:
            denom = dm_safe + gamma * d_j0[None, :]
        
        eta = (prizes_safe ** alpha) / (denom ** beta)
        eta[eta <= 0] = 1e-9
        eta[~np.isfinite(eta)] = 1e-9
        return eta
    
    # Candidate parameter sets to evaluate
    candidate_params = [
        # Baseline
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.0, 'form': 0},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'form': 0},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 2.0, 'form': 0},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.5, 'form': 0},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.5, 'form': 0},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.0, 'form': 1},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'form': 1},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 2.0, 'form': 1},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.5, 'form': 1},
        # Different alpha
        {'alpha': 0.5, 'beta': 1.0, 'gamma': 1.0, 'form': 0},
        {'alpha': 2.0, 'beta': 1.0, 'gamma': 1.0, 'form': 0},
        {'alpha': 1.5, 'beta': 1.0, 'gamma': 1.0, 'form': 0},
        # Different beta
        {'alpha': 1.0, 'beta': 0.5, 'gamma': 1.0, 'form': 0},
        {'alpha': 1.0, 'beta': 2.0, 'gamma': 1.0, 'form': 0},
        {'alpha': 1.0, 'beta': 1.5, 'gamma': 1.0, 'form': 0},
        # Combined
        {'alpha': 1.5, 'beta': 1.5, 'gamma': 1.0, 'form': 0},
        {'alpha': 1.5, 'beta': 1.0, 'gamma': 2.0, 'form': 0},
        {'alpha': 1.0, 'beta': 1.5, 'gamma': 2.0, 'form': 0},
        {'alpha': 2.0, 'beta': 1.5, 'gamma': 1.5, 'form': 0},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 3.0, 'form': 0},
    ]
    
    eval_seeds = [0, 1, 2]
    
    best_score = -1
    best_eta = None
    
    for params in candidate_params:
        if time.time() - start_time > time_limit - 15:
            break
        
        eta = make_eta(params)
        
        scores = []
        for seed in eval_seeds:
            if time.time() - start_time > time_limit - 10:
                break
            s = run_aco(eta, seed)
            scores.append(s)
        
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_eta = eta.copy()
    
    # If we still have time, try a local search around the best parameters
    if best_eta is not None and time.time() - start_time < time_limit - 30:
        # Find which params gave best_score
        # Re-run to identify best params (or track them)
        # For simplicity, do a small random perturbation search
        best_params = {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'form': 0}
        
        # Track best params from the candidate loop
        # Re-evaluate to find best params
        for params in candidate_params:
            eta = make_eta(params)
            scores = []
            for seed in eval_seeds:
                if time.time() - start_time > time_limit - 20:
                    break
                s = run_aco(eta, seed)
                scores.append(s)
            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_eta = eta.copy()
                    best_params = params
        
        # Local random search around best_params
        np.random.seed(42)
        for _ in range(20):
            if time.time() - start_time > time_limit - 15:
                break
            
            new_params = {
                'alpha': max(0.1, best_params['alpha'] + np.random.normal(0, 0.3)),
                'beta': max(0.1, best_params['beta'] + np.random.normal(0, 0.3)),
                'gamma': max(0.0, best_params['gamma'] + np.random.normal(0, 0.5)),
                'form': best_params['form'] if np.random.rand() > 0.3 else (1 - best_params['form'])
            }
            
            eta = make_eta(new_params)
            scores = []
            for seed in eval_seeds:
                if time.time() - start_time > time_limit - 10:
                    break
                s = run_aco(eta, seed)
                scores.append(s)
            
            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_eta = eta.copy()
                    best_params = new_params
    
    # Final safety check
    if best_eta is None or best_eta.shape != dm.shape or not np.all(np.isfinite(best_eta)):
        # Fallback: simple prize/distance heuristic
        dm_safe = np.maximum(dm, 1e-9)
        best_eta = prizes_arr[None, :] / dm_safe
        best_eta[best_eta <= 0] = 1e-9
        best_eta[~np.isfinite(best_eta)] = 1e-9
    
    return best_eta

# EVOLVE-BLOCK-END
