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

def _ffd_assignment(demands, capacity):
    """
    Perform First-Fit-Decreasing bin packing.
    Returns: list of bins, where each bin is a list of item indices.
    """
    n = len(demands)
    # Sort items by size descending
    sorted_indices = np.argsort(-demands)
    
    bins = []  # list of (load, [item_indices])
    
    for idx in sorted_indices:
        placed = False
        for bin_idx, (load, items) in enumerate(bins):
            if load + demands[idx] <= capacity:
                bins[bin_idx] = (load + demands[idx], items + [idx])
                placed = True
                break
        if not placed:
            bins.append((demands[idx], [idx]))
    
    return [items for load, items in bins]

def _build_ffd_heuristic(demands, capacity, strength=100.0, base=1.0):
    """
    Build a heuristic matrix based on FFD bin assignments.
    Items in the same FFD bin get high attractiveness; others get base value.
    """
    n = len(demands)
    d = np.array(demands)
    
    # Get FFD assignment
    ffd_bins = _ffd_assignment(d, capacity)
    
    # Build item-to-bin mapping
    item_to_bin = {}
    for bin_idx, items in enumerate(ffd_bins):
        for item in items:
            item_to_bin[item] = bin_idx
    
    # Build heuristic matrix
    H = np.full((n, n), base)
    
    # For each pair of items in the same FFD bin, set high attractiveness
    for bin_idx, items in enumerate(ffd_bins):
        for i in items:
            for j in items:
                if i != j:
                    # Check if feasible
                    if d[i] + d[j] <= capacity:
                        H[i, j] = strength
                    else:
                        H[i, j] = 1e-9
    
    # Ensure non-positive values become 1e-9
    H[H <= 0] = 1e-9
    
    return H

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 290 # Safety margin for 330s kill, 40s buffer
    
    n = len(demands)
    d = np.array(demands)
    
    # Precompute pair sums and ratios
    pair_sum = d[:, None] + d[None, :]
    feasible_mask = pair_sum <= capacity
    ratio = pair_sum / capacity
    waste = capacity - pair_sum
    
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
    
    # Candidate 3: FFD-based heuristic (strong)
    h3 = _build_ffd_heuristic(d, capacity, strength=100.0, base=1.0)
    candidates.append(h3)
    
    # Candidate 4: FFD-based heuristic (very strong)
    h4 = _build_ffd_heuristic(d, capacity, strength=1000.0, base=0.01)
    candidates.append(h4)
    
    # Candidate 5: FFD-based heuristic (extreme)
    h5 = _build_ffd_heuristic(d, capacity, strength=10000.0, base=0.001)
    candidates.append(h5)
    
    # Candidate 6: Inverse Waste (1 / (capacity - pair_sum + eps))
    h6 = np.where(feasible_mask, 1.0 / (waste + 1.0), 1e-9)
    candidates.append(h6)
    
    # Candidate 7: Best Fit Style (1 / waste^2)
    h7 = np.where(feasible_mask, 1.0 / (waste**2 + 1.0), 1e-9)
    candidates.append(h7)
    
    # Candidate 8: Exp(-waste)
    h8 = np.where(feasible_mask, np.exp(-waste / 5.0), 1e-9)
    candidates.append(h8)
    
    # Candidate 9: Stronger Inverse Waste (1 / waste^3)
    h9 = np.where(feasible_mask, 1.0 / (waste**3 + 1.0), 1e-9)
    candidates.append(h9)
    
    # Candidate 10: Exp(-waste / 2.0) - Stronger exponential decay
    h10 = np.where(feasible_mask, np.exp(-waste / 2.0), 1e-9)
    candidates.append(h10)
    
    # Candidate 11: Ratio^2 * (1 / waste) - Hybrid
    h11 = np.where(feasible_mask, ratio**2 * (1.0 / (waste + 1.0)), 1e-9)
    candidates.append(h11)
    
    # Candidate 12: (1/waste)^4 - Very strong best-fit
    h12 = np.where(feasible_mask, 1.0 / (waste**4 + 1.0), 1e-9)
    candidates.append(h12)
    
    # Candidate 13: FFD + waste hybrid (same bin gets bonus scaled by inverse waste)
    ffd_bins = _ffd_assignment(d, capacity)
    item_to_bin = {}
    for bin_idx, items in enumerate(ffd_bins):
        for item in items:
            item_to_bin[item] = bin_idx
    
    h13 = np.full((n, n), 1e-9)
    for i in range(n):
        for j in range(n):
            if i == j:
                h13[i, j] = 1e-9
                continue
            if d[i] + d[j] <= capacity:
                if item_to_bin[i] == item_to_bin[j]:
                    # Same FFD bin: high base + inverse waste bonus
                    w = capacity - d[i] - d[j]
                    h13[i, j] = 50.0 + 10.0 / (w + 1.0)
                else:
                    # Different bins: lower value based on waste
                    w = capacity - d[i] - d[j]
                    h13[i, j] = 1.0 / (w + 1.0)
    
    candidates.append(h13)
    
    best_score = float('inf')
    best_h = h0 # Fallback to baseline
    
    # Use a few seeds to evaluate robustness
    seeds = [42, 123, 7]
    
    lower_bound = np.ceil(np.sum(d) / capacity)
    
    for idx, h in enumerate(candidates):
        # Check time
        if time.time() > deadline - 30: 
            break
            
        # Evaluate candidate
        scores = []
        for seed in seeds:
            score = simulate_aco(d, capacity, h, seed)
            scores.append(score)
        
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_h = h
            
        # If we hit a very low score (e.g., close to theoretical lower bound), we can stop early
        if best_score <= lower_bound + 1:
            break

    return best_h
