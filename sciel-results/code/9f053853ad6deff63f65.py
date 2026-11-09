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
    # Conservative time limit to ensure we return within 300s
    time_limit = 280.0
    
    # Pre-compute matrices
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    best_score = np.inf
    best_heuristic = None
    
    # Use faster, lower-fidelity ACO for tuning
    def evaluate_heuristic(base):
        score = _run_aco_fast(d, capacity, base, n_ants=5, n_iters=5, seed=42)
        return score
    
    def try_heuristic(base):
        nonlocal best_score, best_heuristic
        base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
        score = evaluate_heuristic(base)
        if score < best_score:
            best_score = score
            best_heuristic = base.copy()
        return score

    # Generate FFD bin structure
    sorted_indices = np.argsort(d)[::-1]
    ffd_bins = []
    current_bin = []
    current_sum = 0.0
    for idx in sorted_indices:
        item_size = d[idx]
        if current_sum + item_size <= capacity:
            current_bin.append(idx)
            current_sum += item_size
        else:
            if current_bin:
                ffd_bins.append(current_bin)
            current_bin = [idx]
            current_sum = item_size
    if current_bin:
        ffd_bins.append(current_bin)
    
    # Create matrices for FFD-based structural heuristics
    # 1. Co-occurrence: items in same FFD bin
    ffd_cooccur = np.zeros((n, n), dtype=bool)
    for b in ffd_bins:
        for i in b:
            for j in b:
                if i != j:
                    ffd_cooccur[i, j] = True
    
    # 2. Adjacency in FFD: items that are consecutive in the same FFD bin
    ffd_adjacent = np.zeros((n, n), dtype=bool)
    for b in ffd_bins:
        for k in range(len(b) - 1):
            i, j = b[k], b[k + 1]
            ffd_adjacent[i, j] = True
            ffd_adjacent[j, i] = True
    
    # 3. Bin-level affinity: how many items share a bin (normalized)
    bin_size = np.zeros(n, dtype=float)
    for b in ffd_bins:
        for i in b:
            bin_size[i] = len(b)
    
    # Structural heuristic family: sparse compatibility based on FFD structure
    # Strategy: high weight for FFD-co-occurring pairs, moderate for adjacent, low for others
    # This creates a landscape that strongly guides ants toward FFD-like solutions
    
    # Variant A: Pure FFD co-occurrence with best-fit tiebreaker
    # High value for co-occurring pairs, scaled by slack
    if time.time() - start_time < time_limit:
        base_a = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
        # Boost co-occurring pairs by a large factor
        boost_a = np.where(ffd_cooccur, 10.0, 1.0)
        base_a = base_a * boost_a
        try_heuristic(base_a)
    
    # Variant B: FFD adjacency with strong emphasis
    if time.time() - start_time < time_limit:
        base_b = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
        boost_b = np.where(ffd_adjacent, 20.0, 1.0)
        base_b = base_b * boost_b
        try_heuristic(base_b)
    
    # Variant C: Combined co-occurrence and adjacency
    if time.time() - start_time < time_limit:
        base_c = np.where(valid, 1.0 / (slack + 1e-9)**1.5, 1e-9)
        boost_c = np.where(ffd_cooccur, 5.0, 1.0) * np.where(ffd_adjacent, 3.0, 1.0)
        base_c = base_c * boost_c
        try_heuristic(base_c)
    
    # Variant D: Bin-size weighted - items in smaller bins get higher priority
    if time.time() - start_time < time_limit:
        # Items in tighter bins (fewer items) are more "valuable"
        bin_weight = 1.0 / (bin_size + 1.0)
        base_d = np.where(valid, bin_weight[:, None] * bin_weight[None, :] / (slack + 1e-9), 1e-9)
        # Boost co-occurring
        boost_d = np.where(ffd_cooccur, 5.0, 1.0)
        base_d = base_d * boost_d
        try_heuristic(base_d)
    
    # Variant E: Pure structural - only FFD co-occurring pairs have meaningful values
    if time.time() - start_time < time_limit:
        # Non-co-occurring pairs get minimal value, co-occurring get slack-based value
        base_e = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
        # Hard gate: non-co-occurring pairs get very low value
        gate_e = np.where(ffd_cooccur, 1.0, 0.01)
        base_e = base_e * gate_e
        try_heuristic(base_e)
    
    # Variant F: Adjacency-only sparse matrix
    if time.time() - start_time < time_limit:
        base_f = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
        gate_f = np.where(ffd_adjacent, 1.0, 0.01)
        base_f = base_f * gate_f
        try_heuristic(base_f)
    
    # Variant G: Hybrid - FFD structure with size-based scaling
    if time.time() - start_time < time_limit:
        # Scale by size ratio: larger items should pair with complementary sizes
        size_ratio = d[:, None] / (d[None, :] + 1e-9)
        base_g = np.where(valid, size_ratio / (slack + 1e-9), 1e-9)
        boost_g = np.where(ffd_cooccur, 8.0, 1.0)
        base_g = base_g * boost_g
        try_heuristic(base_g)
    
    # Variant H: Inverse bin count - items that appear in fewer alternative bins
    if time.time() - start_time < time_limit:
        # Count how many bins each item could potentially fit in (approximate)
        # This is complex, so use a simpler proxy: bin_size
        base_h = np.where(valid, 1.0 / (slack + 1e-9)**2, 1e-9)
        # Items in larger bins have more flexibility, so de-emphasize them
        flex_weight = 1.0 / (bin_size + 1.0)
        base_h = base_h * flex_weight[:, None] * flex_weight[None, :]
        boost_h = np.where(ffd_cooccur, 5.0, 1.0)
        base_h = base_h * boost_h
        try_heuristic(base_h)
    
    # If we have significant time left, try more variants
    if time.time() - start_time < time_limit - 100:
        # Variant I: Exponential slack with FFD gate
        if time.time() - start_time < time_limit:
            base_i = np.where(valid, np.exp(-slack / (capacity * 0.1)), 1e-9)
            gate_i = np.where(ffd_cooccur, 1.0, 0.05)
            base_i = base_i * gate_i
            try_heuristic(base_i)
        
        # Variant J: Quadratic slack with adjacency boost
        if time.time() - start_time < time_limit:
            base_j = np.where(valid, 1.0 / (slack + 1e-9)**2, 1e-9)
            boost_j = np.where(ffd_adjacent, 15.0, 1.0)
            base_j = base_j * boost_j
            try_heuristic(base_j)
        
        # Variant K: Combined all FFD signals
        if time.time() - start_time < time_limit:
            base_k = np.where(valid, 1.0 / (slack + 1e-9)**1.5, 1e-9)
            # Multi-level boost
            boost_k = np.ones((n, n))
            boost_k[ffd_cooccur] *= 4.0
            boost_k[ffd_adjacent] *= 3.0
            base_k = base_k * boost_k
            try_heuristic(base_k)
        
        # Variant L: Size-complementarity with FFD
        if time.time() - start_time < time_limit:
            # Prefer pairs where sizes are complementary (one large, one small)
            size_diff = np.abs(d[:, None] - d[None, :])
            base_l = np.where(valid, size_diff / (slack + 1e-9), 1e-9)
            boost_l = np.where(ffd_cooccur, 6.0, 1.0)
            base_l = base_l * boost_l
            try_heuristic(base_l)
    
    # Final fallback if nothing worked
    if best_heuristic is None:
        base = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
        best_heuristic = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
    
    return best_heuristic
