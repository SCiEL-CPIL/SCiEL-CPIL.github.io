import numpy as np
import time

def _fast_aco_eval(demands, capacity, heuristic, n_ants=5, n_iters=10, seed=42):
    """
    Fast evaluation of the heuristic by running a truncated ACO simulation.
    Returns the best number of bins found.
    """
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # Pheromone initialization
    tau = np.ones((n, n))
    
    best_overall = np.inf
    rng = np.random.default_rng(seed)
    
    for it in range(n_iters):
        min_bins_in_iter = np.inf
        
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            current_item = -1
            current_fill = 0.0
            
            # Place a random fitting item
            j = rng.integers(0, n)
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            while True:
                remaining = capacity - current_fill
                # Candidates: not used and d[j] <= remaining
                mask = (~used) & (d <= remaining)
                
                if not np.any(mask):
                    # Open new bin
                    bins += 1
                    unused_indices = np.where(~used)[0]
                    if len(unused_indices) == 0:
                        break
                    j = unused_indices[rng.integers(0, len(unused_indices))]
                    used[j] = True
                    current_fill = d[j]
                    current_item = j
                    continue
                else:
                    # Get probabilities
                    row_tau = tau[current_item, :]
                    row_heur = heuristic[current_item, :]
                    probs = row_tau * row_heur
                    probs[~mask] = 0.0
                    
                    total_prob = np.sum(probs)
                    if total_prob <= 0:
                        candidates = np.where(mask)[0]
                        j = candidates[rng.integers(0, len(candidates))]
                    else:
                        probs = probs / total_prob
                        cum_probs = np.cumsum(probs)
                        r = rng.random()
                        idx = np.searchsorted(cum_probs, r)
                        if idx >= n:
                            idx = n - 1
                        j = idx
                        
                    used[j] = True
                    current_fill += d[j]
                    current_item = j
            
        min_bins_in_iter = min(min_bins_in_iter, bins)
        
        # Update pheromone
        # "every entry gains 1/(fewest bins of that iteration)"
        tau = tau * 0.9 + (1.0 / min_bins_in_iter)
        
        if min_bins_in_iter < best_overall:
            best_overall = min_bins_in_iter
            
    return best_overall

def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    start_time = time.time()
    time_limit = 280.0 # Safety margin
    
    # Pre-compute pair sums and slack
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    best_score = np.inf
    best_alpha = 1.0
    
    # Search over alpha in H_ij = 1 / slack^alpha
    # Extended range: 0.5 to 8.0 to capture strong preferences for tight fits
    alphas = np.linspace(0.5, 8.0, 15)
    
    # Use more seeds for a smoother objective surface
    seeds = [42, 123, 7]
    
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
            
        # Construct Heuristic
        base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
        base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
        
        scores = []
        for seed in seeds:
            score = _fast_aco_eval(d, capacity, base, n_ants=5, n_iters=10, seed=seed)
            scores.append(score)
        
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_alpha = alpha
            
    # Refine around the best alpha with a tighter grid
    if time.time() - start_time < time_limit - 15.0:
        center = best_alpha
        span = max(0.3, 0.25 * center)
        fine_alphas = np.linspace(max(0.1, center - span), center + span, 7)
        for alpha in fine_alphas:
            if time.time() - start_time > time_limit:
                break
            if alpha <= 0:
                alpha = 0.1
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
            
            scores = []
            for seed in seeds:
                score = _fast_aco_eval(d, capacity, base, n_ants=5, n_iters=10, seed=seed)
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha
                
    # Final polish: very fine search around the best alpha found
    if time.time() - start_time < time_limit - 5.0:
        center = best_alpha
        tiny_span = 0.1
        ultra_fine = np.linspace(max(0.1, center - tiny_span), center + tiny_span, 5)
        for alpha in ultra_fine:
            if time.time() - start_time > time_limit:
                break
            if alpha <= 0:
                alpha = 0.1
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
            
            scores = []
            for seed in seeds:
                score = _fast_aco_eval(d, capacity, base, n_ants=5, n_iters=10, seed=seed)
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha
                
    # Construct final heuristic with best_alpha
    final_heuristic = np.where(valid, 1.0 / (slack + 1e-9)**best_alpha, 1e-9)
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), final_heuristic, 1e-9)
    
    return final_heuristic
