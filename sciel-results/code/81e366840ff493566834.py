import numpy as np
import time

def _simulate_fast(items, capacity, w, eps, alpha=0.0, beta=0.0):
    """
    Fast simulation for a given scoring parameter w, eps, alpha, beta.
    - w: weight on tightness (minimize remaining)
    - eps: weight on index (prefer earlier bins)
    - alpha: weight on matching expected next item size
    - beta: weight on fragmentation (variance of remainders)
    """
    max_bins = len(items) + 1
    bins = np.zeros(max_bins, dtype=np.float64)
    n = 0
    
    for item in items:
        if n == 0:
            bins[0] = capacity - item
            n = 1
            continue
            
        # Vectorized operations on active bins
        rem = bins[:n] - item
        valid = rem >= 0
        
        if not np.any(valid):
            bins[n] = capacity - item
            n += 1
            continue
            
        indices = np.arange(n, dtype=np.float64)
        
        # Base score: tightness + index penalty
        scores = -w * rem - eps * indices
        
        # Alpha: match expected next item size
        # We estimate expected next as the mean of current remainders
        # This is a heuristic for "what size of item is likely to fit next"
        if alpha > 0.0 and n > 0:
            # Mean of current remainders (including the one we're about to fill? No, current state)
            # Actually, we should look at the distribution of remaining spaces in OTHER bins?
            # Or just the global mean of remainders?
            # Let's use the mean of the remainders of the bins we are NOT choosing?
            # That's hard to compute in vectorized form without knowing the choice.
            # Let's stick to the global mean of remainders of all open bins before placing.
            mean_rem = np.mean(bins[:n])
            # Penalize deviation from mean_rem
            next_match = -alpha * ((rem - mean_rem) ** 2) / (2.0 * capacity * capacity)
            scores += next_match
            
        # Beta: Fragmentation penalty
        # We want to minimize the variance of the remainders.
        # If we place in bin i, the new remainder is rem[i].
        # The variance of the set of remainders {rem[0], ..., rem[i-1], rem[i+1], ..., rem[n-1], rem[i]}
        # is roughly proportional to the sum of squared deviations from the new mean.
        # Approximation: Prefer bins where the new remainder is close to the mean of the other remainders.
        # This keeps the distribution tight.
        if beta > 0.0 and n > 1:
            # Calculate mean of all remainders
            total_rem = np.sum(bins[:n])
            mean_rem = total_rem / n
            
            # If we pick bin i, new mean is (total_rem - bins[i] + rem[i]) / n = (total_rem - item) / n
            # New variance is sum((new_rems - new_mean)^2)
            # This is complex to vectorize perfectly.
            # Simpler heuristic: Prefer bins where rem[i] is close to the mean of the *other* bins.
            # mean_others[i] = (total_rem - bins[i]) / (n-1)
            # Score += -beta * (rem[i] - mean_others[i])^2
            # This encourages placing items such that the resulting gap is similar to the gaps in other bins.
            
            if n > 1:
                # total_rem is sum of bins[:n]
                # bins[:n] is the current remaining capacity
                # rem[i] = bins[i] - item
                # mean_others[i] = (total_rem - bins[i]) / (n-1)
                # diff[i] = rem[i] - mean_others[i]
                # We want to minimize diff^2
                # scores += -beta * diff^2
                
                # To avoid division by zero or issues, we can just use a normalized version
                # Let's compute the vector of mean_others
                # Note: bins[:n] are the current capacities
                # rem = bins[:n] - item
                # total_rem = np.sum(bins[:n])
                
                # mean_others[i] = (total_rem - bins[i]) / (n-1)
                # diff[i] = (bins[i] - item) - (total_rem - bins[i]) / (n-1)
                # diff[i] = (bins[i] - item) - (total_rem/(n-1) - bins[i]/(n-1))
                # diff[i] = bins[i] * (1 + 1/(n-1)) - item - total_rem/(n-1)
                # diff[i] = bins[i] * (n/(n-1)) - (item * (n-1) + total_rem) / (n-1)
                
                # Let's just compute it directly for clarity in the simulation loop, 
                # but vectorized it might be slow if done naively.
                # However, n is small (<1000).
                
                # Let's try a simpler proxy: 
                # Standard deviation of remainders is a measure of fragmentation.
                # We want to choose the bin that minimizes the new standard deviation.
                # This is equivalent to minimizing the sum of squared distances from the new mean.
                
                # Let's use the "distance to mean of others" heuristic as it's a good proxy for keeping the distribution tight.
                
                # Compute mean_others
                # To do this vectorized:
                # mean_others = (total_rem - bins[:n]) / (n-1)
                # diff = rem - mean_others
                # frag_penalty = -beta * (diff ** 2) / (capacity * capacity)
                
                if n > 1:
                    mean_others = (total_rem - bins[:n]) / (n - 1)
                    diff = rem - mean_others
                    frag_penalty = -beta * (diff ** 2) / (capacity * capacity)
                    scores += frag_penalty
                    # Ensure invalid bins are still -inf
                    scores[~valid] = -np.inf
                else:
                    scores[~valid] = -np.inf
            else:
                scores[~valid] = -np.inf
        else:
            scores[~valid] = -np.inf
            
        best_idx = int(np.argmax(scores))
        bins[best_idx] -= item
        
    return n

def _generate_weibull(n, capacity, scale, shape, seed):
    """Generate Weibull-distributed items scaled to capacity."""
    rng = np.random.default_rng(seed)
    raw = rng.weibull(shape, size=n)
    raw = raw / raw.max()
    items = raw * capacity * 0.9
    return items

def _l1_lower_bound(items, capacity):
    """Compute L1 lower bound on number of bins."""
    total = sum(items)
    max_item = max(items)
    return max(np.ceil(total / capacity), np.ceil(max_item / capacity))

def _score_excess(n_bins, l1_bound):
    """Compute excess over L1 lower bound in percent."""
    if l1_bound == 0:
        return 0.0
    return (n_bins - l1_bound) / l1_bound * 100.0

def _optimize_parameters():
    """Optimize scoring parameters via simulation."""
    start = time.time()
    deadline = start + 110  # Safety margin for 120s limit (or 150s)
    
    # Generate test sequences
    sequences = []
    for s in range(5):
        items = _generate_weibull(1000, 100, 1.0, 1.5, seed=s)
        sequences.append(items)
        
    best_score = float('inf')
    best_w = 1.0
    best_eps = 0.0
    best_alpha = 0.0
    best_beta = 0.0
    
    # Coarse grid search
    # w: tightness
    # eps: index
    # alpha: next-item match
    # beta: fragmentation
    
    w_vals = np.arange(0.0, 3.1, 0.5)
    eps_vals = [0.0, 0.01]
    alpha_vals = [0.0, 1.0, 2.0]
    beta_vals = [0.0, 0.5, 1.0, 2.0]
    
    for w in w_vals:
        for eps in eps_vals:
            for alpha in alpha_vals:
                for beta in beta_vals:
                    if time.time() > deadline - 30:
                        break
                    if time.time() > deadline - 30:
                        break
                    if time.time() > deadline - 30:
                        break
                        
                    total_excess = 0
                    for items in sequences:
                        n_bins = _simulate_fast(items, 100, w, eps, alpha, beta)
                        l1 = _l1_lower_bound(items, 100)
                        total_excess += _score_excess(n_bins, l1)
                    
                    avg_excess = total_excess / 5
                    if avg_excess < best_score:
                        best_score = avg_excess
                        best_w = w
                        best_eps = eps
                        best_alpha = alpha
                        best_beta = beta
                        
    # Local search refinement
    if time.time() < deadline - 10:
        # Refine around best values
        for delta_w in np.linspace(-0.2, 0.2, 5):
            w_test = best_w + delta_w
            for eps_test in [best_eps, 0.0, 0.01]:
                for delta_a in np.linspace(-0.5, 0.5, 3):
                    alpha_test = max(0.0, best_alpha + delta_a)
                    for delta_b in np.linspace(-0.5, 0.5, 3):
                        beta_test = max(0.0, best_beta + delta_b)
                        if time.time() > deadline - 5:
                            break
                        total_excess = 0
                        for items in sequences:
                            n_bins = _simulate_fast(items, 100, w_test, eps_test, alpha_test, beta_test)
                            l1 = _l1_lower_bound(items, 100)
                            total_excess += _score_excess(n_bins, l1)
                        
                        avg_excess = total_excess / 5
                        if avg_excess < best_score:
                            best_score = avg_excess
                            best_w = w_test
                            best_eps = eps_test
                            best_alpha = alpha_test
                            best_beta = beta_test

    return best_w, best_eps, best_alpha, best_beta

# Run optimization at import time
_W, _EPS, _ALPHA, _BETA = _optimize_parameters()

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score bins: higher score = better choice."""
    remaining = bins - item
    valid_mask = remaining >= 0
    n = len(bins)
    
    if n == 0:
        return np.array([])
        
    indices = np.arange(n, dtype=np.float64)
    
    # Base score
    score = -_W * remaining - _EPS * indices
    
    # Alpha: match expected next item size
    # Use mean of current remainders as proxy
    if _ALPHA > 0.0 and n > 0:
        mean_rem = np.mean(bins)
        next_match = -_ALPHA * ((remaining - mean_rem) ** 2) / (2.0 * 100.0 * 100.0)
        score += next_match
        
    # Beta: Fragmentation penalty
    if _BETA > 0.0 and n > 1:
        total_rem = np.sum(bins)
        mean_others = (total_rem - bins) / (n - 1)
        diff = remaining - mean_others
        frag_penalty = -_BETA * (diff ** 2) / (100.0 * 100.0)
        score += frag_penalty
        
    score[~valid_mask] = -np.inf
    
    return score
