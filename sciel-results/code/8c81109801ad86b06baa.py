import numpy as np
import time

def simulate_aco_fast(demands, capacity, H, seed, n_ants=5, n_iters=10):
    """Faster simulation for candidate evaluation with fewer ants/iterations"""
    np.random.seed(seed)
    n = len(demands)
    tau = np.ones((n, n))
    best_overall = float('inf')
    
    for _ in range(n_iters):
        bins_counts = []
        for _ in range(n_ants):
            remaining = list(range(n))
            bins = 0
            current_bin_items = []
            current_load = 0
            last_item = -1
            
            while remaining:
                if not current_bin_items:
                    idx = np.random.randint(len(remaining))
                    i = remaining[idx]
                    remaining.pop(idx)
                    current_bin_items.append(i)
                    current_load = demands[i]
                    last_item = i
                    bins += 1
                else:
                    feasible_indices = []
                    weights = []
                    for k, j in enumerate(remaining):
                        if current_load + demands[j] <= capacity:
                            feasible_indices.append(k)
                            w = tau[last_item, j] * H[last_item, j]
                            if w <= 0: 
                                w = 1e-9
                            weights.append(w)
                    
                    if not feasible_indices:
                        current_bin_items = []
                        current_load = 0
                        last_item = -1
                        continue
                    
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
        
        tau *= 0.9
        tau += 1.0 / best_this_iter
    
    return best_overall

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
    waste = capacity - pair_sum
    h5 = np.where(feasible_mask, 1.0 / (waste + 1.0), 1e-9)
    candidates.append(h5)
    
    # Candidate 6: Best Fit Style (1 / waste^2)
    h6 = np.where(feasible_mask, 1.0 / (waste**2 + 1.0), 1e-9)
    candidates.append(h6)
    
    # Candidate 7: Exp(-waste)
    h7 = np.where(feasible_mask, np.exp(-waste / 5.0), 1e-9)
    candidates.append(h7)
    
    # FFD-based heuristics
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
    
    # Candidate 8: FFD-based sparse compatibility
    h_ffd = np.full((n, n), 1e-9)
    for b in bins:
        for i in b:
            for j in b:
                if i != j and feasible_mask[i, j]:
                    h_ffd[i, j] = 1.0
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
    
    # New: First Fit Decreasing (FFD) with waste-aware scaling
    # Candidate 10: FFD pairs weighted by inverse waste
    h10 = np.full((n, n), 1e-9)
    for b in bins:
        for i in b:
            for j in b:
                if i != j and feasible_mask[i, j]:
                    h10[i, j] = 1.0 / (waste[i, j] + 1.0)
    candidates.append(h10)
    
    # New: Hybrid - blend ratio and inverse waste smoothly
    # Candidate 11: Geometric mean of ratio and inverse waste
    h11 = np.where(feasible_mask, np.sqrt(ratio * (1.0 / (waste + 1.0))), 1e-9)
    candidates.append(h11)
    
    # New: Weighted sum of multiple strategies
    # Candidate 12: 0.5*ratio + 0.5*exp(-waste/5)
    h12 = np.where(feasible_mask, 0.5 * ratio + 0.5 * np.exp(-waste / 5.0), 1e-9)
    candidates.append(h12)
    
    # New: Size compatibility - prefer pairs where sizes are similar
    # Candidate 13: Exp(-|d_i - d_j| / scale)
    size_diff = np.abs(d[:, None] - d[None, :])
    h13 = np.where(feasible_mask, np.exp(-size_diff / 20.0), 1e-9)
    candidates.append(h13)
    
    # New: Combined - ratio * size_compatibility
    # Candidate 14: ratio * exp(-size_diff / 20)
    h14 = np.where(feasible_mask, ratio * np.exp(-size_diff / 20.0), 1e-9)
    candidates.append(h14)
    
    # New: Soft FFD - use bin index difference as penalty
    # Candidate 15: FFD with smooth penalty for cross-bin pairs
    bin_idx = np.full(n, -1)
    for b, bin_items in enumerate(bins):
        for item in bin_items:
            bin_idx[item] = b
    bin_diff = np.abs(bin_idx[:, None] - bin_idx[None, :])
    h15 = np.where(feasible_mask, np.exp(-bin_diff / 3.0) * ratio, 1e-9)
    candidates.append(h15)
    
    # New: Best-fit inspired - strongly favor pairs that fill bins tightly
    # Candidate 16: 1/(waste^3 + 1) for very tight fits
    h16 = np.where(feasible_mask, 1.0 / (waste**3 + 1.0), 1e-9)
    candidates.append(h16)
    
    # New: Logarithmic waste penalty
    # Candidate 17: exp(-log(waste + 1) / 2)
    h17 = np.where(feasible_mask, np.exp(-np.log(waste + 1.0) / 2.0), 1e-9)
    candidates.append(h17)
    
    best_score = float('inf')
    best_h = h0 # Fallback to baseline
    
    # Use a few seeds to evaluate robustness
    seeds = [42, 123, 7]
    
    # First pass: fast evaluation with fewer ants/iterations
    for idx, h in enumerate(candidates):
        # Check time
        if time.time() > deadline - 60: 
            break
            
        # Fast evaluation
        scores = []
        for seed in seeds:
            score = simulate_aco_fast(d, capacity, h, seed, n_ants=5, n_iters=10)
            scores.append(score)
        
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_h = h
            
        # If we hit a very low score, we can stop early
        lower_bound = np.ceil(np.sum(d) / capacity)
        if best_score <= lower_bound + 1:
            break
    
    # Second pass: refine top candidates with full simulation
    # Keep top 3 candidates from fast evaluation for detailed evaluation
    if time.time() < deadline - 30:
        # Re-evaluate the best candidate found with full simulation
        full_scores = []
        for seed in seeds:
            score = simulate_aco(d, capacity, best_h, seed)
            full_scores.append(score)
        full_avg = np.mean(full_scores)
        
        # Also try a few other promising candidates with full simulation
        # Pick candidates that scored well in fast evaluation
        candidate_scores = []
        for idx, h in enumerate(candidates):
            fast_scores = []
            for seed in seeds[:1]:  # Use 1 seed for speed
                score = simulate_aco_fast(d, capacity, h, seed, n_ants=5, n_iters=10)
                fast_scores.append(score)
            candidate_scores.append((np.mean(fast_scores), idx))
        
        # Sort by fast score, take top 3
        candidate_scores.sort()
        top_candidates = [candidate_scores[i][1] for i in range(min(3, len(candidate_scores)))]
        
        for idx in top_candidates:
            if time.time() > deadline - 15:
                break
            h = candidates[idx]
            scores = []
            for seed in seeds:
                score = simulate_aco(d, capacity, h, seed)
                scores.append(score)
            avg_score = np.mean(scores)
            if avg_score < full_avg:
                full_avg = avg_score
                best_h = h
    
    return best_h
