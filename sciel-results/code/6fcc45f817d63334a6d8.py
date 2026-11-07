"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _simulate_aco(demands, capacity, H, n_ants=20, n_iters=30, seed=42):
    """Run ACO with given heuristic matrix, return fewest bins found."""
    rng = np.random.RandomState(seed)
    n = len(demands)
    demands = np.array(demands, dtype=float)
    capacity = float(capacity)
    
    # Pheromone matrix
    tau = np.ones((n, n))
    
    best_bins = float('inf')
    
    for it in range(n_iters):
        iteration_best = float('inf')
        
        for ant in range(n_ants):
            bins = 0
            used = np.zeros(n, dtype=bool)
            current = None
            
            while not used.all():
                # Find fitting items
                if current is None:
                    # Pick random fitting item to start bin
                    fitting = np.where(~used)[0]
                    if len(fitting) == 0:
                        break
                    current = fitting[rng.randint(len(fitting))]
                    used[current] = True
                    bins += 1
                else:
                    # Find items that fit after current
                    remaining_capacity = capacity - demands[current]
                    fitting = np.where((~used) & (demands <= remaining_capacity))[0]
                    
                    if len(fitting) == 0:
                        # Open new bin
                        current = None
                        continue
                    
                    # Compute probabilities
                    probs = tau[current, fitting] * H[current, fitting]
                    total = probs.sum()
                    if total <= 0 or not np.isfinite(total):
                        probs = np.ones(len(fitting))
                        total = probs.sum()
                    
                    probs /= total
                    next_idx = rng.choice(fitting, p=probs)
                    used[next_idx] = True
                    current = next_idx
            
            if bins < iteration_best:
                iteration_best = bins
        
        if iteration_best < best_bins:
            best_bins = iteration_best
        
        # Update pheromone
        tau *= 0.9
        tau += 1.0 / iteration_best
    
    return best_bins


def _compute_heuristic(demands, capacity, alpha, beta, gamma, delta):
    """Compute heuristic with given blend parameters.
    
    alpha: weight on inverse gap (1/(cap - pair_sum))
    beta: weight on next item size (demands[j])
    gamma: weight on pair sum / capacity (fill ratio)
    delta: weight on current item size (demands[i])
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= capacity
    
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = capacity - pair_sums
        h_inv_gap = np.where(fit_mask, 1.0 / (gap + 1e-9), 1e-9)
        h_next_size = np.where(fit_mask, W[None, :], 1e-9)
        h_fill = np.where(fit_mask, pair_sums / capacity, 1e-9)
        h_curr_size = np.where(fit_mask, W[:, None], 1e-9)
    
    H = alpha * h_inv_gap + beta * h_next_size + gamma * h_fill + delta * h_curr_size
    
    # Ensure non-negative and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H


def heuristics(demands, capacity):
    W = np.array(demands, dtype=float)
    n = len(W)
    
    start_time = time.time()
    deadline = start_time + 280  # 20s margin for safety
    
    # Grid search over blend parameters
    best_H = None
    best_score = float('inf')
    
    # Candidate parameter combinations
    # Focus on strong inverse gap signal (alpha high) with small blends
    alphas = [1.0, 2.0, 5.0]
    betas = [0.0, 0.1, 0.3, 0.5]
    gammas = [0.0, 0.1, 0.3]
    deltas = [0.0, 0.1, 0.3]
    
    # Generate candidate list, prioritize high alpha
    candidates = []
    for a in alphas:
        for b in betas:
            for g in gammas:
                for d in deltas:
                    candidates.append((a, b, g, d))
    
    # Sort by expected quality: high alpha first
    candidates.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    
    # Evaluate candidates with time budget
    for (a, b, g, d) in candidates:
        if time.time() > deadline:
            break
        
        H = _compute_heuristic(W, capacity, a, b, g, d)
        
        # Quick evaluation with reduced ACO
        score = _simulate_aco(W, capacity, H, n_ants=10, n_iters=15, seed=42)
        
        if score < best_score:
            best_score = score
            best_H = H
    
    # If we have time, do a finer search around the best
    if best_H is not None and time.time() < deadline - 30:
        # Refine around best parameters
        best_params = None
        for (a, b, g, d) in candidates:
            H_test = _compute_heuristic(W, capacity, a, b, g, d)
            if np.array_equal(H_test, best_H):
                best_params = (a, b, g, d)
                break
        
        if best_params is not None:
            a0, b0, g0, d0 = best_params
            # Fine-tune search
            fine_candidates = []
            for da in [-0.5, -0.2, 0.2, 0.5]:
                for db in [-0.1, 0.0, 0.1]:
                    for dg in [-0.1, 0.0, 0.1]:
                        for dd in [-0.1, 0.0, 0.1]:
                            na = max(0.1, a0 + da)
                            nb = max(0.0, b0 + db)
                            ng = max(0.0, g0 + dg)
                            nd = max(0.0, d0 + dd)
                            fine_candidates.append((na, nb, ng, nd))
            
            for (a, b, g, d) in fine_candidates:
                if time.time() > deadline:
                    break
                
                H = _compute_heuristic(W, capacity, a, b, g, d)
                score = _simulate_aco(W, capacity, H, n_ants=10, n_iters=15, seed=42)
                
                if score < best_score:
                    best_score = score
                    best_H = H
    
    # Final: if no valid heuristic found, use default
    if best_H is None:
        pair = W[:, None] + W[None, :]
        best_H = np.where(pair <= capacity, pair / capacity, 1e-9)
    
    return best_H
# EVOLVE-BLOCK-END
