import numpy as np
import time

def _run_aco_fast(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation using precomputed candidate lists and vectorized operations.
    Returns the minimum number of bins used by any ant.
    Optimized for speed: uses local variables and minimal overhead.
    """
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # Precompute: for each item i, which items j can follow i (d[i] + d[j] <= capacity)
    pair_sum = d[:, None] + d[None, :]
    valid_mask = pair_sum <= capacity
    
    # Precompute candidate indices for each item
    candidates_list = [np.where(valid_mask[i])[0] for i in range(n)]
    
    # Pheromone initialization
    tau = np.ones((n, n))
    
    best_overall = np.inf
    rng = np.random.default_rng(seed)
    
    for it in range(n_iters):
        min_bins_in_iter = n + 1 # Max possible is n
        
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            # Open first bin - place random item
            j = rng.integers(0, n)
            used[j] = True
            current_item = j
            
            # Place remaining items
            while True:
                cands = candidates_list[current_item]
                if len(cands) == 0:
                    break
                valid_cands = cands[~used[cands]]
                if len(valid_cands) == 0:
                    bins += 1
                    
                    unused_indices = np.where(~used)[0]
                    if len(unused_indices) == 0:
                        break
                    j = unused_indices[rng.integers(0, len(unused_indices))]
                    used[j] = True
                    current_item = j
                    continue
                
                # Compute probabilities
                row_tau = tau[current_item, valid_cands]
                row_heur = heuristic[current_item, valid_cands]
                probs = row_tau * row_heur
                
                total_prob = np.sum(probs)
                if total_prob <= 0:
                    j = valid_cands[rng.integers(0, len(valid_cands))]
                else:
                    probs /= total_prob
                    r = rng.random()
                    cum_probs = np.cumsum(probs)
                    idx = np.searchsorted(cum_probs, r)
                    if idx >= len(valid_cands):
                        idx = len(valid_cands) - 1
                    j = valid_cands[idx]
                
                used[j] = True
                current_item = j

            if bins < min_bins_in_iter:
                min_bins_in_iter = bins
        
        if min_bins_in_iter < best_overall:
            best_overall = min_bins_in_iter
            
        tau = tau * 0.9 + (1.0 / min_bins_in_iter)
        
    return best_overall

def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    start_time = time.time()
    # Conservative time limit to ensure we return within 330s
    time_limit = 310.0
    
    # Pre-compute matrices
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    # Compute FFD assignment for each item
    # Sort items by size descending, then assign to bin with least remaining capacity that fits
    sorted_indices = np.argsort(-d)
    bin_remaining = []
    item_bin = np.zeros(n, dtype=int)
    bin_count = 0
    
    for idx in sorted_indices:
        placed = False
        for b in range(bin_count):
            if bin_remaining[b] >= d[idx]:
                # Check if this is the best fit (least remaining)
                best_b = b
                for b2 in range(b+1, bin_count):
                    if bin_remaining[b2] >= d[idx] and bin_remaining[b2] < bin_remaining[best_b]:
                        best_b = b2
                bin_remaining[best_b] -= d[idx]
                item_bin[idx] = best_b
                placed = True
                break
        if not placed:
            bin_remaining.append(capacity - d[idx])
            item_bin[idx] = bin_count
            bin_count += 1
    
    # Build FFD adjacency matrix: high value if items are in same bin
    ffd_adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and item_bin[i] == item_bin[j]:
                ffd_adj[i, j] = 1.0
    
    best_score = np.inf
    best_heuristic = None
    
    # Use faster, lower-fidelity ACO for tuning
    # n_ants=10, n_iters=10 is ~1/9th the cost of 20x30
    def evaluate_heuristic(base):
        score = _run_aco_fast(d, capacity, base, n_ants=10, n_iters=10, seed=42)
        return score
    
    def try_heuristic(base):
        nonlocal best_score, best_heuristic
        base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
        score = evaluate_heuristic(base)
        if score < best_score:
            best_score = score
            best_heuristic = base.copy()
        return score
    
    # Focus on the most promising family: Best-Fit (1/slack^alpha)
    # Knowledge suggests alpha=2 is good. Test a range around it.
    alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
        base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
        try_heuristic(base)
        
    # If we have time, test FFD-adjacency heuristic
    if time.time() - start_time < time_limit - 60:
        # FFD adjacency: prefer items from the same FFD bin
        base = np.where(valid, ffd_adj + 0.1, 1e-9)
        try_heuristic(base)
        
        # Hybrid: best-fit + FFD adjacency
        best_fit = np.where(valid, 1.0 / (slack + 1e-9)**2.0, 1e-9)
        base = np.where(valid, best_fit + ffd_adj, 1e-9)
        try_heuristic(base)
        
        # Another hybrid with different weight
        base = np.where(valid, best_fit + 0.5 * ffd_adj, 1e-9)
        try_heuristic(base)
        
        # Target gap (0.8)
        target = capacity * 0.8
        dist = np.abs(pair_sum - target)
        base = np.where(valid, 1.0 / (dist + 1e-9), 1e-9)
        try_heuristic(base)

    if best_heuristic is None:
        # Fallback to standard best-fit
        base = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
        best_heuristic = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
    
    return best_heuristic
