"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 15.0  # seconds before the 330s kill limit

def _time_budget_remaining():
    return (330.0 - _DEADLINE_MARGIN) - (time.time() - _START_TIME)

def _compute_ideal_partners(d, capacity):
    """
    For each item, find its ideal partner via two-pointer sweep on sorted demands.
    Returns an array ideal_partner[i] = index of best complementary item for item i,
    and ideal_gap[i] = gap when paired with that ideal partner.
    """
    n = len(d)
    sorted_idx = np.argsort(d)
    sorted_d = d[sorted_idx]
    
    ideal_partner = np.zeros(n, dtype=int)
    ideal_gap = np.zeros(n, dtype=float)
    
    # Two-pointer: for each item in sorted order, find the largest item that fits with it
    # Since we want pairs that sum close to capacity, for each sorted item k,
    # the ideal partner is the largest item j such that sorted_d[k] + sorted_d[j] <= capacity
    # Use binary search or two-pointer
    
    for k in range(n):
        # Find largest j such that sorted_d[k] + sorted_d[j] <= capacity
        # j can be at most n-1, but we want the one closest to capacity
        target = capacity - sorted_d[k]
        # Find rightmost index where sorted_d <= target
        # Use searchsorted
        pos = np.searchsorted(sorted_d, target, side='right') - 1
        if pos >= 0 and pos < n:
            j_sorted = pos
        else:
            # No valid partner found (shouldn't happen since all items <= 99 and cap=150)
            j_sorted = 0
        
        # Map back to original indices
        ideal_partner[sorted_idx[k]] = sorted_idx[j_sorted]
        ideal_gap[sorted_idx[k]] = capacity - (sorted_d[k] + sorted_d[j_sorted])
    
    return ideal_partner, ideal_gap

def _complementarity_heuristic(d, capacity, sigma=10.0, base_weight=1.0, complement_weight=5.0):
    """
    Build a heuristic based on complementarity to ideal partners.
    
    H[i,j] = base_weight + complement_weight * exp(-|d[i]+d[j] - capacity| / sigma)
            * exp(-rank_distance(i,j) / tau)
    
    where rank_distance measures how far j is from i's ideal partner in sorted order.
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    
    # Compute gaps: how far from capacity
    gaps = np.abs(pair_sums - capacity)
    
    # Fit quality: closer to capacity is better (for fitting pairs)
    fit_quality = np.exp(-gaps / sigma)
    
    # Complementarity: how close is j to i's ideal partner?
    sorted_idx = np.argsort(d)
    rank = np.zeros(n, dtype=int)
    for i, s in enumerate(sorted_idx):
        rank[s] = i
    
    # For each pair (i, j), compute the rank distance from j to i's ideal partner
    ideal_partner = np.zeros(n, dtype=int)
    for k in range(n):
        target = capacity - d[sorted_idx[k]]
        pos = np.searchsorted(d[sorted_idx], target, side='right') - 1
        if pos >= 0 and pos < n:
            ideal_partner[sorted_idx[k]] = sorted_idx[pos]
        else:
            ideal_partner[sorted_idx[k]] = sorted_idx[0]
    
    # Rank distance matrix
    ideal_rank = rank[ideal_partner]  # ideal_rank[i] = rank of i's ideal partner
    rank_diff = np.abs(rank[None, :] - ideal_rank[:, None])  # rank_diff[i,j] = |rank[j] - rank[ideal_partner[i]]|
    
    tau = 5.0  # temperature for rank distance penalty
    complement_score = np.exp(-rank_diff / tau)
    
    # Combine
    h = base_weight + complement_weight * fit_quality * complement_score
    
    # Mask non-fitting pairs
    h = np.where(fits, h, 1e-9)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h

def _fast_proxy_score(d, capacity, heuristic, n_ants=5, seed=42):
    """
    Fast greedy simulation using the heuristic.
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.RandomState(seed)
    
    best_bins = float('inf')
    
    for _ in range(n_ants):
        placed = np.zeros(n, dtype=bool)
        bins_used = 0
        current_bin_sum = 0.0
        last_item = -1
        
        idx = rng.randint(n)
        placed[idx] = True
        current_bin_sum = d[idx]
        last_item = idx
        bins_used = 1
        
        while not np.all(placed):
            remaining_space = capacity - current_bin_sum
            fits_mask = (~placed) & (d <= remaining_space)
            
            if not np.any(fits_mask):
                bins_used += 1
                available = np.where(~placed)[0]
                idx = available[rng.randint(len(available))]
                placed[idx] = True
                current_bin_sum = d[idx]
                last_item = idx
                continue
            
            row_he = heuristic[last_item, :]
            scores = np.where(fits_mask, row_he, 0.0)
            idx = np.argmax(scores)
            
            if scores[idx] == 0:
                candidates = np.where(fits_mask)[0]
                idx = candidates[rng.randint(len(candidates))]
            
            placed[idx] = True
            current_bin_sum += d[idx]
            last_item = idx
            
        if bins_used < best_bins:
            best_bins = bins_used
            
    return best_bins

def _medium_aco_score(d, capacity, heuristic, n_ants=10, n_iterations=15, seed=42):
    """
    Medium-fidelity ACO simulation.
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.RandomState(seed)
    
    pheromone = np.ones((n, n))
    best_bins_ever = float('inf')
    
    for _ in range(n_iterations):
        best_bins_this_iter = float('inf')
        
        for _ in range(n_ants):
            placed = np.zeros(n, dtype=bool)
            bins_used = 0
            current_bin_sum = 0.0
            last_item = -1
            
            idx = rng.randint(n)
            placed[idx] = True
            current_bin_sum = d[idx]
            last_item = idx
            bins_used = 1
            
            while not np.all(placed):
                remaining_space = capacity - current_bin_sum
                fits_mask = (~placed) & (d <= remaining_space)
                
                if not np.any(fits_mask):
                    bins_used += 1
                    available = np.where(~placed)[0]
                    idx = available[rng.randint(len(available))]
                    placed[idx] = True
                    current_bin_sum = d[idx]
                    last_item = idx
                    continue
                
                row_ph = pheromone[last_item, :]
                row_he = heuristic[last_item, :]
                
                scores = row_ph * row_he
                scores = np.where(fits_mask, scores, 0.0)
                
                total_score = np.sum(scores)
                if total_score == 0:
                    candidates = np.where(fits_mask)[0]
                    idx = candidates[rng.randint(len(candidates))]
                else:
                    probs = scores / total_score
                    idx = rng.choice(n, p=probs)
                    
                placed[idx] = True
                current_bin_sum += d[idx]
                last_item = idx
                
        if bins_used < best_bins_this_iter:
            best_bins_this_iter = bins_used
            
        if best_bins_this_iter < best_bins_ever:
            best_bins_ever = best_bins_this_iter
            
        pheromone *= 0.9
        pheromone += 1.0 / best_bins_this_iter
        
    return best_bins_ever

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness using complementarity-based heuristic.
    Combines fit quality (closeness to capacity) with structural complementarity
    (proximity to ideal partner in sorted order).
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Precompute pair sums and fits
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    gaps = np.abs(pair_sums - capacity)
    
    # Compute ideal partners via two-pointer
    sorted_idx = np.argsort(d)
    sorted_d = d[sorted_idx]
    rank = np.zeros(n, dtype=int)
    for i, s in enumerate(sorted_idx):
        rank[s] = i
    
    ideal_partner = np.zeros(n, dtype=int)
    for k in range(n):
        target = capacity - sorted_d[k]
        pos = np.searchsorted(sorted_d, target, side='right') - 1
        if pos >= 0 and pos < n:
            ideal_partner[sorted_idx[k]] = sorted_idx[pos]
        else:
            ideal_partner[sorted_idx[k]] = sorted_idx[0]
    
    ideal_rank = rank[ideal_partner]
    
    # Search over sigma (fit temperature) and tau (rank distance temperature)
    best_score = float('inf')
    best_heuristic = None
    
    sigmas = [5.0, 8.0, 10.0, 15.0, 20.0, 30.0]
    taus = [3.0, 5.0, 7.0, 10.0, 15.0]
    base_weights = [0.5, 1.0, 2.0]
    complement_weights = [3.0, 5.0, 8.0, 12.0]
    
    candidates = []
    for sw in sigmas:
        for tw in taus:
            for bw in base_weights:
                for cw in complement_weights:
                    candidates.append((sw, tw, bw, cw))
    
    # Shuffle
    rng_search = np.random.RandomState(42)
    rng_search.shuffle(candidates)
    
    # Stage 1: Fast proxy evaluation
    proxy_results = []
    for sigma, tau, bw, cw in candidates:
        if _time_budget_remaining() < 25.0:
            break
        
        # Build heuristic
        fit_quality = np.exp(-gaps / sigma)
        rank_diff = np.abs(rank[None, :] - ideal_rank[:, None])
        complement_score = np.exp(-rank_diff / tau)
        
        h = bw + cw * fit_quality * complement_score
        h = np.where(fits, h, 1e-9)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
        score = _fast_proxy_score(d, capacity, h, n_ants=5, seed=42)
        proxy_results.append((score, sigma, tau, bw, cw, h))
    
    if not proxy_results:
        # Fallback
        sigma, tau, bw, cw = 10.0, 5.0, 1.0, 5.0
        fit_quality = np.exp(-gaps / sigma)
        rank_diff = np.abs(rank[None, :] - ideal_rank[:, None])
        complement_score = np.exp(-rank_diff / tau)
        h = bw + cw * fit_quality * complement_score
        h = np.where(fits, h, 1e-9)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        return h
    
    # Sort by proxy score
    proxy_results.sort(key=lambda x: x[0])
    
    # Stage 2: Medium ACO for top candidates
    top_candidates = proxy_results[:8]
    
    for score, sigma, tau, bw, cw, h in top_candidates:
        if _time_budget_remaining() < 12.0:
            break
        
        med_score = _medium_aco_score(d, capacity, h, n_ants=10, n_iterations=15, seed=42)
        
        if med_score < best_score:
            best_score = med_score
            best_heuristic = h.copy()
    
    if best_heuristic is None:
        best_heuristic = top_candidates[0][5]
        
    return best_heuristic

# EVOLVE-BLOCK-END
