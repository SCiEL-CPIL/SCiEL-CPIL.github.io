import numpy as np
import time

def _run_aco_fast(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation using precomputed candidate lists and vectorized operations.
    Returns the minimum number of bins used by any ant.
    """
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # Precompute: for each item i, which items j can follow i (d[i] + d[j] <= capacity)
    # This is the "valid" mask
    pair_sum = d[:, None] + d[None, :]
    valid_mask = pair_sum <= capacity
    
    # Precompute candidate indices for each item (items that can follow it)
    # For efficiency, we'll use numpy arrays
    candidates_list = [np.where(valid_mask[i])[0] for i in range(n)]
    
    # Pheromone initialization
    tau = np.ones((n, n))
    
    best_overall = np.inf
    rng = np.random.default_rng(seed)
    
    for it in range(n_iters):
        ant_bins = []
        
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            current_item = -1
            current_fill = 0.0
            
            # Open first bin - place random item
            j = rng.integers(0, n)
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            # Place remaining items
            while True:
                remaining = capacity - current_fill
                # Get candidates that are unused and fit
                cands = candidates_list[current_item]
                if len(cands) == 0:
                    break
                # Filter to unused items that fit
                valid_cands = cands[~used[cands]]
                if len(valid_cands) == 0:
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
                
                # Compute probabilities for valid candidates
                row_tau = tau[current_item, valid_cands]
                row_heur = heuristic[current_item, valid_cands]
                probs = row_tau * row_heur
                
                total_prob = np.sum(probs)
                if total_prob <= 0:
                    j = valid_cands[rng.integers(0, len(valid_cands))]
                else:
                    probs = probs / total_prob
                    r = rng.random()
                    # Use searchsorted on cumulative sum
                    cum_probs = np.cumsum(probs)
                    idx = np.searchsorted(cum_probs, r)
                    if idx >= len(valid_cands):
                        idx = len(valid_cands) - 1
                    j = valid_cands[idx]
                
                used[j] = True
                current_fill += d[j]
                current_item = j

            ant_bins.append(bins)
            
        min_bins = min(ant_bins)
        if min_bins < best_overall:
            best_overall = min_bins
            
        # Pheromone update
        tau = tau * 0.9 + (1.0 / min_bins)
        
    return best_overall

def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    start_time = time.time()
    time_limit = 320.0  # Leave margin before 330s limit
    
    # Pre-compute slack matrix
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    best_score = np.inf
    best_heuristic = None
    
    # Use the fast ACO runner
    def evaluate_heuristic(base):
        scores = []
        for seed in [42, 123, 456]:
            score = _run_aco_fast(d, capacity, base, n_ants=20, n_iters=30, seed=seed)
            scores.append(score)
        return np.mean(scores)
    
    # Candidate heuristics to evaluate
    # We prioritize "Best Fit" variants (minimizing slack) as per knowledge hint
    # 1. Power law: 1 / (slack + eps)^alpha
    # 2. Exponential decay: exp(-slack / sigma)
    
    candidate_heuristics = []
    
    # Power law alphas: Standard, Sharp, Very Sharp
    alphas = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    for alpha in alphas:
        if alpha == 0:
            base = np.where(valid, 1.0, 1e-9)
        else:
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
        base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
        candidate_heuristics.append(base)
        
    # Exponential decay sigmas
    sigmas = [5.0, 10.0, 20.0, 50.0, 100.0]
    for sigma in sigmas:
        # exp(-slack/sigma) gives high value for small slack
        base = np.where(valid, np.exp(-slack / sigma), 1e-9)
        base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
        candidate_heuristics.append(base)
        
    # Evaluate candidates
    # Sort candidates by a proxy score? Hard to do without running ACO.
    # We will just iterate. If time runs out, we return the best found so far.
    # To be safe, we evaluate the most likely good ones first (alpha=1, 2, 4 and sigma=10, 20)
    
    # Reorder to prioritize known good regions
    # alpha=1 (index 1), alpha=2 (index 2), alpha=4 (index 3)
    # sigma=10 (index 7), sigma=20 (index 8)
    # We'll just iterate through the list, but maybe swap a few?
    # Actually, the list is small (12 items). 12 * 3 seeds * ~0.1s = 3.6s.
    # We have 320s. We can afford to try more.
    
    # Let's add more variations if time permits later, but first evaluate these.
    
    for i, base in enumerate(candidate_heuristics):
        if time.time() - start_time > time_limit:
            break
        
        avg_score = evaluate_heuristic(base)
        
        if avg_score < best_score:
            best_score = avg_score
            best_heuristic = base.copy()
            
    # If we have significant time left, try more granular search around the best alpha/sigma
    # or try mixed heuristics.
    if time.time() - start_time < time_limit - 10.0:
        # Try mixed heuristics: mix * best_fit + (1-mix) * worst_fit
        # Worst fit: slack^1.0
        worst_fit = np.where(valid, (slack + 1e-9)**1.0, 1e-9)
        worst_fit_norm = (worst_fit - worst_fit.min()) / (worst_fit.max() - worst_fit.min() + 1e-12)
        
        # Best fit alpha=2 normalized
        bf_2 = np.where(valid, 1.0 / (slack + 1e-9)**2.0, 1e-9)
        bf_2_norm = (bf_2 - bf_2.min()) / (bf_2.max() - bf_2.min() + 1e-12)
        
        mixes = [0.1, 0.3, 0.5, 0.7, 0.9]
        for mix in mixes:
            if time.time() - start_time > time_limit:
                break
            base = np.where(valid, mix * bf_2_norm + (1-mix) * worst_fit_norm, 1e-9)
            base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
            
            avg_score = evaluate_heuristic(base)
            if avg_score < best_score:
                best_score = avg_score
                best_heuristic = base.copy()
                
    # If still time, try even sharper alphas
    if time.time() - start_time < time_limit - 10.0:
        for alpha in [64.0, 128.0]:
            if time.time() - start_time > time_limit:
                break
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
            
            avg_score = evaluate_heuristic(base)
            if avg_score < best_score:
                best_score = avg_score
                best_heuristic = base.copy()
                
    if best_heuristic is None:
        # Fallback to Best Fit (alpha=1)
        base = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
        best_heuristic = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
    
    return best_heuristic
