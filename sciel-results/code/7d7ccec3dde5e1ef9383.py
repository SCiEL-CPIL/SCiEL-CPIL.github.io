import numpy as np
import time

def simulate_aco(demands, capacity, H, seed):
    np.random.seed(seed)
    n = len(demands)
    tau = np.ones((n, n))
    best_overall = float('inf')
    
    for _ in range(30):
        bins_counts = []
        for _ in range(20):
            remaining = list(range(n))
            bins = 0
            current_bin_items = []
            current_load = 0
            last_item = -1
            
            while remaining:
                if not current_bin_items:
                    # Random fitting item
                    idx = np.random.randint(len(remaining))
                    i = remaining[idx]
                    remaining.pop(idx)
                    current_bin_items.append(i)
                    current_load = demands[i]
                    last_item = i
                    bins += 1
                else:
                    # Find feasible j
                    feasible_indices = []
                    weights = []
                    # Iterate through remaining items
                    for k, j in enumerate(remaining):
                        if current_load + demands[j] <= capacity:
                            feasible_indices.append(k)
                            # prob proportional to tau[last_item, j] * H[last_item, j]
                            w = tau[last_item, j] * H[last_item, j]
                            if w <= 0: 
                                w = 1e-9
                            weights.append(w)
                    
                    if not feasible_indices:
                        # Open new bin
                        current_bin_items = []
                        current_load = 0
                        last_item = -1
                        continue
                    
                    # Select j
                    total_w = sum(weights)
                    r = np.random.rand() * total_w
                    cumulative = 0
                    chosen_k = feasible_indices[0]
                    for k_idx in range(len(feasible_indices)):
                        cumulative += weights[k_idx]
                        if r <= cumulative:
                            chosen_k = feasible_indices[k_idx]
                            break
                    
                    j = remaining[chosen_k]
                    remaining.pop(chosen_k)
                    current_bin_items.append(j)
                    current_load += demands[j]
                    last_item = j
            
            bins_counts.append(bins)
        
        best_this_iter = min(bins_counts)
        if best_this_iter < best_overall:
            best_overall = best_this_iter
        
        # Update Pheromone
        tau *= 0.9
        tau += 1.0 / best_this_iter
        
    return best_overall

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 290 # Safety margin for 330s kill, 40s buffer
    
    n = len(demands)
    d = np.array(demands)
    
    # Precompute pair sums and ratios
    pair_sum = d[:, None] + d[None, :]
    feasible_mask = pair_sum <= capacity
    ratio = pair_sum / capacity
    
    candidates = []
    
    # Candidate 0: Baseline (Ratio^1)
    h0 = np.where(feasible_mask, ratio, 1e-9)
    candidates.append(h0)
    
    # Candidate 1: Ratio^2
    h1 = np.where(feasible_mask, ratio**2, 1e-9)
    candidates.append(h1)
    
    # Candidate 2: Ratio^4
    h2 = np.where(feasible_mask, ratio**4, 1e-9)
    candidates.append(h2)
    
    # Candidate 3: Ratio^8
    h3 = np.where(feasible_mask, ratio**8, 1e-9)
    candidates.append(h3)
    
    # Candidate 4: Ratio^16
    h4 = np.where(feasible_mask, ratio**16, 1e-9)
    candidates.append(h4)
    
    # Candidate 5: Inverse Waste (1 / (capacity - pair_sum + eps))
    # This favors tight fits but is unbounded if waste is small.
    # Adding eps=1 prevents division by zero and limits max value.
    waste = capacity - pair_sum
    h5 = np.where(feasible_mask, 1.0 / (waste + 1.0), 1e-9)
    candidates.append(h5)
    
    # Candidate 6: Best Fit Style (1 / waste^2)
    h6 = np.where(feasible_mask, 1.0 / (waste**2 + 1.0), 1e-9)
    candidates.append(h6)
    
    # Candidate 7: Exp(-waste)
    h7 = np.where(feasible_mask, np.exp(-waste / 5.0), 1e-9)
    candidates.append(h7)
    
    # Candidate 8: Residual Spectrum - reward residuals that match remaining item distribution
    # For each pair (i,j), compute residual = capacity - d[i] - d[j]
    # The attractiveness should depend on how well this residual can accommodate
    # typical remaining items. We use the median and quantiles of the demand distribution.
    median_demand = np.median(d)
    q25 = np.percentile(d, 25)
    q75 = np.percentile(d, 75)
    
    # Residual spectrum: for a given residual r, how many items could fit?
    # We want residuals that are large enough to fit at least one more typical item
    # but not so large that we waste space on items that don't fit well.
    
    # Score each pair based on residual matching the distribution
    # A good residual is one where:
    # 1. It's at least the median demand (so we can fit a typical item)
    # 2. It's not excessively large (so we don't waste space)
    
    # Use a Gaussian-like scoring around the ideal residual
    # Ideal residual: around median demand, with penalty for being too small or too large
    ideal_residual = median_demand
    sigma = (q75 - q25) / 2.0 + 1.0  # Spread based on IQR
    
    residual = capacity - pair_sum
    # Score: exp(-((residual - ideal)^2 / (2*sigma^2))) for feasible pairs
    residual_score = np.exp(-((residual - ideal_residual)**2) / (2 * sigma**2))
    h8 = np.where(feasible_mask, residual_score, 1e-9)
    candidates.append(h8)
    
    # Candidate 9: Residual spectrum with stronger preference for tight fits
    # Use a skewed Gaussian: penalize large residuals more than small ones
    # (since we can always add more items to fill space, but can't remove)
    skew_factor = 1.5
    residual_score_skew = np.exp(-((residual - ideal_residual * skew_factor)**2) / (2 * sigma**2))
    # Also boost if residual is close to median
    h9 = np.where(feasible_mask, residual_score_skew * (1.0 + 0.5 * np.exp(-((residual - median_demand)**2) / (2 * sigma**2))), 1e-9)
    candidates.append(h9)
    
    # Candidate 10: Combined residual spectrum and inverse waste
    # Blend the residual matching with tight-fit preference
    h10 = np.where(feasible_mask, 0.6 * residual_score + 0.4 * (1.0 / (waste + 1.0)), 1e-9)
    candidates.append(h10)
    
    # Candidate 11: Residual spectrum with quantile-based scoring
    # Score based on which quantile of the demand distribution the residual falls in
    # Residuals that match the 50th-75th percentile of demands are ideal
    def residual_quantile_score(res_arr, d_arr):
        scores = np.zeros_like(res_arr)
        for i in range(len(d_arr)):
            for j in range(len(d_arr)):
                r = res_arr[i, j]
                if r < 0:
                    scores[i, j] = 1e-9
                else:
                    # Find what fraction of items have demand <= r
                    frac = np.sum(d_arr <= r) / len(d_arr)
                    # Ideal fraction: 0.5 to 0.75 (median to 75th percentile)
                    # Score peaks at 0.625
                    ideal_frac = 0.625
                    scores[i, j] = np.exp(-((frac - ideal_frac)**2) / 0.05)
        return scores
    
    h11 = residual_quantile_score(residual, d)
    h11 = np.where(feasible_mask, h11, 1e-9)
    candidates.append(h11)
    
    # Candidate 12: FFD-adjacency with residual spectrum blending
    # Use FFD bin assignments but modulate by residual spectrum
    sorted_indices = np.argsort(-d)
    bins = []
    bin_loads = []
    for idx in sorted_indices:
        placed = False
        for b in range(len(bins)):
            if bin_loads[b] + d[idx] <= capacity:
                bins[b].append(idx)
                bin_loads[b] += d[idx]
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_loads.append(d[idx])
    
    h_ffd = np.full((n, n), 1e-9)
    for b in bins:
        for i in b:
            for j in b:
                if i != j and feasible_mask[i, j]:
                    h_ffd[i, j] = 1.0
    
    # Blend FFD with residual spectrum
    h12 = np.where(h_ffd > 0.5, 50.0 * residual_score + 10.0, np.where(feasible_mask, residual_score, 1e-9))
    candidates.append(h12)
    
    # Candidate 13: FFD with ratio scaling within bins (original)
    h13 = np.full((n, n), 1e-9)
    for b in bins:
        for i in b:
            for j in b:
                if i != j and feasible_mask[i, j]:
                    h13[i, j] = 50.0 + ratio[i, j]
    candidates.append(h13)
    
    # Candidate 14: Pure FFD (original strong FFD)
    h14 = np.where(h_ffd > 0.5, 100.0, np.where(feasible_mask, ratio, 1e-9))
    candidates.append(h14)
    
    # Candidate 15: Residual spectrum with adaptive ideal based on bin packing context
    # The ideal residual depends on how many items are already in the bin
    # Since we don't have that info in the static heuristic, use a conservative ideal
    # that balances between fitting one more item and not wasting too much space
    conservative_ideal = (median_demand + q75) / 2.0
    sigma2 = (q75 - q25) / 3.0 + 1.0
    residual_score2 = np.exp(-((residual - conservative_ideal)**2) / (2 * sigma2**2))
    h15 = np.where(feasible_mask, residual_score2, 1e-9)
    candidates.append(h15)
    
    # Candidate 16: Multi-scale residual spectrum
    # Combine scores from multiple scales of the demand distribution
    h16 = np.zeros((n, n))
    for scale in [0.5, 1.0, 1.5, 2.0]:
        ideal_s = median_demand * scale
        sigma_s = (q75 - q25) / 2.0 * scale + 1.0
        score_s = np.exp(-((residual - ideal_s)**2) / (2 * sigma_s**2))
        h16 += score_s
    h16 = np.where(feasible_mask, h16 / 4.0, 1e-9)
    candidates.append(h16)
    
    # Candidate 17: Residual spectrum with explicit penalty for residuals that can't fit any remaining item
    # If residual < min(demand), it's useless
    min_demand = np.min(d)
    h17 = np.where(feasible_mask, residual_score * np.where(residual >= min_demand, 1.0, 0.1), 1e-9)
    candidates.append(h17)
    
    # Candidate 18: Residual spectrum with bonus for residuals that match specific common demands
    # Find the most common demand values and give bonus if residual matches them
    demand_counts = np.bincount(d.astype(int))
    common_demands = np.where(demand_counts > 1)[0]
    h18 = residual_score.copy()
    for cd in common_demands:
        # Bonus if residual is close to this common demand
        bonus = np.exp(-((residual - cd)**2) / 10.0) * (demand_counts[cd] / len(d))
        h18 += bonus
    h18 = np.where(feasible_mask, h18, 1e-9)
    candidates.append(h18)
    
    # Candidate 19: Residual spectrum with exponential decay for very large residuals
    # Large residuals waste space, so penalize them exponentially
    max_useful_residual = q75 * 1.5
    h19 = np.where(feasible_mask, residual_score * np.exp(-np.maximum(0, residual - max_useful_residual) / 10.0), 1e-9)
    candidates.append(h19)
    
    # Candidate 8: FFD-based sparse compatibility
    # Sort items by size descending
    sorted_indices = np.argsort(-d)
    bins = []
    bin_loads = []
    for idx in sorted_indices:
        placed = False
        for b in range(len(bins)):
            if bin_loads[b] + d[idx] <= capacity:
                bins[b].append(idx)
                bin_loads[b] += d[idx]
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_loads.append(d[idx])
    
    # Build heuristic matrix based on FFD bin assignments
    h_ffd = np.full((n, n), 1e-9)
    for b in bins:
        for i in b:
            for j in b:
                if i != j and feasible_mask[i, j]:
                    h_ffd[i, j] = 1.0
    
    # Blend FFD with ratio-based heuristic
    # We want to strongly encourage FFD pairs but allow some flexibility
    # Use a high value for FFD pairs, lower for others
    h8 = np.where(h_ffd > 0.5, 100.0, np.where(feasible_mask, ratio, 1e-9))
    candidates.append(h8)
    
    # Candidate 9: FFD with ratio scaling within bins
    h9 = np.full((n, n), 1e-9)
    for b in bins:
        for i in b:
            for j in b:
                if i != j and feasible_mask[i, j]:
                    h9[i, j] = 50.0 + ratio[i, j]
    candidates.append(h9)
    
    best_score = float('inf')
    best_h = h0 # Fallback to baseline
    
    # Use a few seeds to evaluate robustness
    seeds = [42, 123, 7]
    
    # First pass: evaluate all candidates with 1 seed for quick screening
    quick_scores = []
    for idx, h in enumerate(candidates):
        if time.time() > deadline - 60:
            break
        score = simulate_aco(d, capacity, h, seeds[0])
        quick_scores.append((idx, score))
    
    # Sort by quick score and take top 8 for detailed evaluation
    quick_scores.sort(key=lambda x: x[1])
    top_indices = [idx for idx, _ in quick_scores[:8]]
    
    for idx in top_indices:
        h = candidates[idx]
        # Check time
        if time.time() > deadline - 30: 
            break
            
        # Evaluate candidate with all seeds
        scores = []
        for seed in seeds:
            score = simulate_aco(d, capacity, h, seed)
            scores.append(score)
        
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_h = h
            
        # If we hit a very low score (e.g., close to theoretical lower bound), we can stop early
        # Lower bound is ceil(sum(demands) / capacity)
        lower_bound = np.ceil(np.sum(d) / capacity)
        if best_score <= lower_bound + 1:
            break

    return best_h
