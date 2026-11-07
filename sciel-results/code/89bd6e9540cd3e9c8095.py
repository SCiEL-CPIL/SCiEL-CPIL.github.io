"""Program for ant colony optimization of offline bin packing with heuristic design."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 20.0  # seconds before the 330s kill limit

def _time_budget_remaining():
    return (330.0 - _DEADLINE_MARGIN) - (time.time() - _START_TIME)

def _greedy_score(d, capacity, heuristic, n_ants=10, seed=42):
    """
    Fast greedy simulation: pick item j that maximizes heuristic[current, j]
    among fitting items. This approximates the ACO when pheromone is uniform.
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
        
        # Open first bin, random start
        idx = rng.randint(n)
        placed[idx] = True
        current_bin_sum = d[idx]
        last_item = idx
        bins_used = 1
        
        while not np.all(placed):
            remaining_space = capacity - current_bin_sum
            fits_mask = (~placed) & (d <= remaining_space)
            
            if not np.any(fits_mask):
                # Open new bin
                bins_used += 1
                available = np.where(~placed)[0]
                idx = available[rng.randint(len(available))]
                placed[idx] = True
                current_bin_sum = d[idx]
                last_item = idx
                continue
            
            # Greedy choice: max heuristic[last_item, j] among fitting j
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

def _aco_score(d, capacity, heuristic, n_ants=20, n_iterations=30, seed=42):
    """
    Full ACO simulation as per task description.
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    
    pheromone = np.ones((n, n))
    rng = np.random.RandomState(seed)
    
    best_bins_ever = float('inf')
    
    for _ in range(n_iterations):
        best_bins_this_iter = float('inf')
        
        for _ in range(n_ants):
            placed = np.zeros(n, dtype=bool)
            bins_used = 0
            current_bin_sum = 0.0
            last_item = -1
            
            # Open a bin, place a random fitting item
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
            
        # Update pheromone
        pheromone *= 0.9
        pheromone += 1.0 / best_bins_this_iter
        
    return best_bins_ever

def _build_heuristic(d, capacity, alpha, beta, gamma, delta):
    """
    Build heuristic matrix from parameters.
    H_ij = ((d_i+d_j)/C)^alpha * (1/(1+gap))^beta * (1 + gamma*(1 - fill)^2 + delta*|d_i - d_j|/C)
    where fill = (d_i+d_j)/C, gap = C - (d_i+d_j)
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    gaps = capacity - pair_sums
    fills = pair_sums / capacity
    
    # Term 1: prefer fuller bins (higher fill)
    term1 = np.power(np.maximum(fills, 1e-6), alpha)
    
    # Term 2: prefer tight fit (small gap)
    term2 = np.power(1.0 / (1.0 + gaps), beta)
    
    # Term 3: bonus for balanced pairs (similar sizes) and penalty for very different
    # Actually, for bin packing, we might want to prefer items that complement
    # Let's use: bonus for making the bin close to full
    # fill^2 encourages packing more tightly
    term3 = 1.0 + gamma * (1.0 - fills)**2
    
    # Term 4: diversity in sizes might help packing
    diff = np.abs(d[:, None] - d[None, :]) / capacity
    term4 = 1.0 + delta * diff
    
    h = term1 * term2 * term3 * term4
    
    # For non-fitting pairs, set to minimum
    h = np.where(fits, h, 1e-9)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h

def _build_complementarity_heuristic(d, capacity, sharpness=5.0, base_power=2.0, tight_bonus=2.0, tight_threshold=5.0):
    """
    Build heuristic based on complementarity score using two-pointer sweep.
    
    Strategy:
    1. Sort demands and for each item, find its "ideal partner" via two-pointer
       (the item that makes the pair sum closest to capacity without exceeding it)
    2. Assign high attractiveness to pairs near these ideal complements
    3. Penalize large deviations from ideal pairing structure
    4. Add tight-fit bonus for pairs with very small waste
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    
    # Sort demands and track original indices
    sorted_indices = np.argsort(d)
    sorted_d = d[sorted_indices]
    
    # For each sorted position, find ideal partner using two-pointer
    # ideal_partner[i] = index j such that d[i]+d[j] is closest to capacity (<= capacity)
    ideal_partner = np.zeros(n, dtype=np.int64)
    ideal_gap = np.full(n, capacity, dtype=np.float64)
    
    # Two-pointer sweep: for each i, find j that minimizes gap = capacity - d[i] - d[j]
    # where d[i] + d[j] <= capacity
    j = n - 1
    for i in range(n):
        # Move j left while d[i] + d[j] > capacity
        while j > i and sorted_d[i] + sorted_d[j] > capacity:
            j -= 1
        if j >= i:
            gap = capacity - sorted_d[i] - sorted_d[j]
            if gap < ideal_gap[sorted_indices[i]]:
                ideal_gap[sorted_indices[i]] = gap
                ideal_partner[sorted_indices[i]] = sorted_indices[j]
        # Also check if staying at current j or moving right gives better
        # Actually, for each i, the best j is the largest j where d[i]+d[j]<=capacity
        # Let's do this more carefully
    
    # Reset and do proper two-pointer
    ideal_partner = np.zeros(n, dtype=np.int64)
    ideal_gap = np.full(n, capacity, dtype=np.float64)
    
    for i in range(n):
        # Binary search for largest j where sorted_d[i] + sorted_d[j] <= capacity
        target = capacity - sorted_d[i]
        # Find rightmost j where sorted_d[j] <= target
        j = np.searchsorted(sorted_d, target, side='right') - 1
        if j >= 0:
            gap = capacity - sorted_d[i] - sorted_d[j]
            if gap < ideal_gap[sorted_indices[i]]:
                ideal_gap[sorted_indices[i]] = gap
                ideal_partner[sorted_indices[i]] = sorted_indices[j]
    
    # Build complementarity score matrix
    # For pair (i, j): score based on how close d[i]+d[j] is to capacity
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    gaps = capacity - pair_sums
    fills = pair_sums / capacity
    
    # Base score: prefer high fill
    base_score = np.power(np.maximum(fills, 1e-6), base_power)
    
    # Complementarity bonus: high score for pairs that are near-ideal complements
    # For item i, its ideal partner is ideal_partner[i]
    # We want to give high scores to pairs (i, j) where j is close to ideal_partner[i]
    # and d[i]+d[j] is close to capacity
    
    # Create a matrix where comp_score[i,j] = 1 if j == ideal_partner[i], else decay
    comp_score = np.ones((n, n))
    for i in range(n):
        ip = ideal_partner[i]
        # Give high score to the ideal partner
        comp_score[i, ip] = 1.0 + tight_bonus
        # Also give moderate scores to nearby items in sorted order
        # Find items with similar "complementarity"
    
    # Better approach: use the gap directly
    # Small gap = good complement
    gap_score = np.exp(-sharpness * gaps / capacity)
    
    # Combine: base * gap_score
    h = base_score * gap_score
    
    # Add explicit tight-fit bonus for very small gaps
    tight_mask = fits & (gaps < tight_threshold)
    h = np.where(tight_mask, h * (1.0 + tight_bonus), h)
    
    # For non-fitting pairs, set to minimum
    h = np.where(fits, h, 1e-9)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h

def _build_heuristic_v2(d, capacity, alpha, beta, gamma):
    """
    Alternative heuristic: emphasize remaining space utilization.
    H_ij = (remaining_after / C)^alpha * (1/(1+gap))^beta * (1 + gamma*fill)
    where remaining_after = C - d_i - d_j, fill = (d_i+d_j)/C
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    gaps = capacity - pair_sums
    fills = pair_sums / capacity
    remaining = gaps / capacity
    
    # Prefer items that leave less space (tighter fit)
    term1 = np.power(np.maximum(1.0 - remaining, 1e-6), alpha)
    
    # Penalize large gaps
    term2 = np.power(1.0 / (1.0 + gaps), beta)
    
    # Bonus for high fill
    term3 = 1.0 + gamma * fills
    
    h = term1 * term2 * term3
    
    h = np.where(fits, h, 1e-9)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    
    Strategy: Search over heuristic parameters using greedy simulation as proxy,
    then validate top candidates with full ACO simulation if time permits.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Precompute useful arrays
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    
    best_score = float('inf')
    best_heuristic = None
    
    # Stage 1: Greedy search over parameter space
    # Parameters: alpha, beta, gamma for _build_heuristic
    n_samples = 120
    rng_search = np.random.RandomState(7)
    
    # Sample from reasonable ranges
    alphas = np.append(rng_search.uniform(0.5, 8.0, n_samples), [1.0, 2.0, 3.0, 4.0, 5.0])
    betas = np.append(rng_search.uniform(0.0, 5.0, n_samples), [0.5, 1.0, 2.0, 3.0])
    gammas = np.append(rng_search.uniform(0.0, 2.0, n_samples), [0.0, 0.5, 1.0, 1.5])
    
    # Create candidate list
    candidates = []
    for a in alphas:
        for b in betas:
            for g in gammas:
                candidates.append((a, b, g))
    
    # Limit total candidates to keep runtime manageable
    max_candidates = 300
    if len(candidates) > max_candidates:
        # Sample uniformly
        indices = rng_search.choice(len(candidates), size=max_candidates, replace=False)
        candidates = [candidates[i] for i in indices]
    
    # Shuffle
    rng_search.shuffle(candidates)
    
    # Also add complementarity-based candidates
    comp_candidates = []
    sharpnesses = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    base_powers = [1.0, 1.5, 2.0, 2.5, 3.0]
    tight_bonuses = [0.5, 1.0, 1.5, 2.0, 3.0]
    tight_thresholds = [3.0, 5.0, 8.0, 10.0]
    
    for s in sharpnesses:
        for bp in base_powers:
            for tb in tight_bonuses:
                for tt in tight_thresholds:
                    comp_candidates.append((s, bp, tb, tt))
    
    # Sample from complementarity candidates
    max_comp = 200
    if len(comp_candidates) > max_comp:
        comp_indices = rng_search.choice(len(comp_candidates), size=max_comp, replace=False)
        comp_candidates = [comp_candidates[i] for i in comp_indices]
    
    # Evaluate with greedy proxy
    proxy_results = []
    
    for i, (alpha, beta, gamma) in enumerate(candidates):
        if _time_budget_remaining() < 30.0:
            break
        
        h = _build_heuristic(d, capacity, alpha, beta, gamma, delta=0.5)
        
        # Quick greedy score
        score = _greedy_score(d, capacity, h, n_ants=8, seed=42)
        proxy_results.append((score, alpha, beta, gamma, h))
    
    # Evaluate complementarity candidates
    for i, (sharpness, base_power, tight_bonus, tight_threshold) in enumerate(comp_candidates):
        if _time_budget_remaining() < 30.0:
            break
        
        h = _build_complementarity_heuristic(d, capacity, sharpness, base_power, tight_bonus, tight_threshold)
        
        # Quick greedy score
        score = _greedy_score(d, capacity, h, n_ants=8, seed=42)
        proxy_results.append((score, 'comp', sharpness, base_power, h))
        
    if not proxy_results:
        # Fallback: use a reasonable default heuristic
        h = _build_heuristic(d, capacity, alpha=2.0, beta=1.0, gamma=0.5, delta=0.5)
        return h
    
    # Sort by greedy score
    proxy_results.sort(key=lambda x: x[0])
    
    # Stage 2: Validate top candidates with ACO
    top_k = min(10, len(proxy_results))
    for result in proxy_results[:top_k]:
        score = result[0]
        h = result[-1]
        if _time_budget_remaining() < 15.0:
            break
        
        aco_score = _aco_score(d, capacity, h, n_ants=20, n_iterations=30, seed=42)
        
        if aco_score < best_score:
            best_score = aco_score
            best_heuristic = h.copy()
            
    # If no ACO validation was done, use best greedy
    if best_heuristic is None:
        best_heuristic = proxy_results[0][4]
        
    # Final safety check
    if not np.all(np.isfinite(best_heuristic)):
        best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
    if not np.all(best_heuristic > 0):
        best_heuristic = np.where(best_heuristic > 0, best_heuristic, 1e-9)
        
    return best_heuristic
# EVOLVE-BLOCK-END
