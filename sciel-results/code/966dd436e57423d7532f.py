import numpy as np
import time
import random
from scipy.optimize import minimize

def _calc_prob_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for a discrete measure.
    Vectorized for speed.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights to probabilities
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Pairwise sums matrix: S[i,j] = s_i + s_j
    S = support[:, None] + support[None, :]
    
    # Weights outer product: W[i,j] = p_i * p_j
    W = p[:, None] * p[None, :]
    
    S_flat = S.flatten()
    W_flat = W.flatten()
    
    total_prob = 0.0
    
    # Pre-sort S_flat and corresponding W_flat for fast prefix sums
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    W_cumsum = np.cumsum(W_sorted)
    W_cumsum_padded = np.concatenate(([0.0], W_cumsum))
    
    for k in range(n):
        threshold = 2.0 * support[k]
        # We need sum_l p_l * (sum_{i,j} p_i p_j I(s_i+s_j < threshold - s_l))
        # Let T_l = threshold - s_l
        T = threshold - support
        
        # Find indices in S_sorted where S_sorted < T
        # np.searchsorted returns index of first element >= T
        # So elements [0:idx] are < T
        indices = np.searchsorted(S_sorted, T, side='left')
        
        # Sum of weights for indices < T
        # W_cumsum_padded[indices] gives sum of first 'indices' elements
        sums_for_l = W_cumsum_padded[indices]
        
        # prob3 = sum_l p_l * sums_for_l
        prob3 = np.dot(p, sums_for_l)
        
        total_prob += p[k] * prob3
        
    return total_prob

def _validate(support, weights):
    if len(support) != len(weights):
        return False
    if len(support) < 1:
        return False
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return False
    if np.any(np.isinf(support)) or np.any(np.isinf(weights)):
        return False
    if np.any(support < -1e-9):
        return False
    if np.any(weights < -1e-9):
        return False
    if np.sum(weights) <= 1e-9:
        return False
    return True

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using a structural random search with local refinement.
    Strategy:
    1. Start with a small random distribution.
    2. Randomly mutate support points and weights.
    3. Occasionally add or remove a point.
    4. Accept moves based on a simulated annealing schedule or simple hill climbing with random restarts.
    5. Use local Nelder-Mead refinement on promising candidates.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    best_prob = -1.0
    best_support = None
    best_weights = None
    
    # Initial candidate: simple 2-point distribution
    current_support = np.array([0.0, 1.0])
    current_weights = np.array([0.5, 0.5])
    current_prob = _calc_prob_fast(current_support, current_weights)
    
    best_support = current_support.copy()
    best_weights = current_weights.copy()
    best_prob = current_prob
    
    # Simulated Annealing / Random Walk parameters
    temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    # Helper to generate a random perturbation
    def perturb(support, weights):
        n = len(support)
        new_support = support.copy()
        new_weights = weights.copy()
        
        # Mutate support: log-normal perturbation for positive values
        # Support values must be >= 0
        for i in range(n):
            if random.random() < 0.5:
                # Perturb
                if new_support[i] > 1e-6:
                    factor = np.exp(random.gauss(0, 0.5))
                    new_support[i] = max(0.0, new_support[i] * factor)
                else:
                    new_support[i] = abs(random.gauss(0, 1.0))
        
        # Sort support to maintain structure (though order doesn't matter for the measure, 
        # it helps in local search stability if we assume sorted input for some heuristics)
        new_support = np.sort(new_support)
        
        # Mutate weights: Dirichlet-like perturbation
        # Add noise and renormalize
        noise = np.abs(np.random.normal(0, 0.1, size=n))
        new_weights = new_weights + noise
        # Ensure non-negative
        new_weights = np.maximum(new_weights, 0.0)
        w_sum = np.sum(new_weights)
        if w_sum > 1e-9:
            new_weights = new_weights / w_sum
        else:
            new_weights = np.ones(n) / n
            
        return new_support, new_weights

    def add_point(support, weights):
        n = len(support)
        if n >= 8: # Cap size to keep n^2 manageable
            return support, weights
        
        # Insert a new point between existing ones or at edges
        # Random position in log-space relative to current range
        min_s = np.min(support)
        max_s = np.max(support)
        
        if max_s == min_s:
            new_s = min_s + abs(random.gauss(0, 1.0))
        else:
            # Log-uniform interpolation
            log_min = np.log(max(min_s, 1e-6))
            log_max = np.log(max(max_s, 1e-6))
            log_new = random.uniform(log_min, log_max)
            new_s = np.exp(log_new)
            
        # Sometimes add a point near 0 or very large
        if random.random() < 0.2:
            new_s = abs(random.gauss(0, 0.5))
        elif random.random() < 0.2:
            new_s = max_s * np.exp(abs(random.gauss(0, 1.0)))
            
        new_support = np.insert(support, random.randint(0, n), new_s)
        new_support = np.sort(new_support)
        
        # Split weight of a random existing point
        idx = random.randint(0, n-1)
        w_split = weights[idx] * random.uniform(0.1, 0.5)
        
        new_weights = weights.copy()
        new_weights[idx] -= w_split
        # Insert new weight
        # Find insertion index in sorted support
        insert_idx = np.searchsorted(new_support, new_s)
        new_weights = np.insert(new_weights, insert_idx, w_split)
        
        # Renormalize
        w_sum = np.sum(new_weights)
        if w_sum > 1e-9:
            new_weights = new_weights / w_sum
        else:
            new_weights = np.ones(len(new_support)) / len(new_support)
            
        return new_support, new_weights

    def remove_point(support, weights):
        n = len(support)
        if n <= 2:
            return support, weights
        
        # Remove a random point
        idx = random.randint(0, n-1)
        new_support = np.delete(support, idx)
        new_weights = np.delete(weights, idx)
        
        # Renormalize
        w_sum = np.sum(new_weights)
        if w_sum > 1e-9:
            new_weights = new_weights / w_sum
        else:
            new_weights = np.ones(len(new_support)) / len(new_support)
            
        return new_support, new_weights

    # Main search loop
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Choose mutation type
        r = random.random()
        if r < 0.7:
            new_support, new_weights = perturb(current_support, current_weights)
        elif r < 0.85 and len(current_support) < 8:
            new_support, new_weights = add_point(current_support, current_weights)
        elif r < 0.95 and len(current_support) > 2:
            new_support, new_weights = remove_point(current_support, current_weights)
        else:
            # Random restart: generate a completely new small distribution
            n_new = random.randint(2, 5)
            new_support = np.sort(np.abs(np.random.normal(0, 2.0, size=n_new)))
            new_support[0] = 0.0 # Anchor at 0 often helps
            new_weights = np.random.dirichlet(np.ones(n_new))
            
        if not _validate(new_support, new_weights):
            continue
            
        new_prob = _calc_prob_fast(new_support, new_weights)
        
        # Acceptance criterion: Simulated Annealing
        delta = new_prob - current_prob
        if delta > 0 or (temp > min_temp and random.random() < np.exp(delta / temp)):
            current_support = new_support
            current_weights = new_weights
            current_prob = new_prob
            
            if current_prob > best_prob:
                best_prob = current_prob
                best_support = current_support.copy()
                best_weights = current_weights.copy()
                
                # If we found a new best, try local refinement
                if time.time() < deadline - 30:
                    # Local refinement using Nelder-Mead
                    # Parameterize: log(support[1:]) and weights
                    # Support[0] is fixed at min(support) or 0? 
                    # Let's optimize all support points > 0 and weights.
                    # To keep it simple, we'll just perturb the current best slightly 
                    # in the raw space, but Nelder-Mead works better on unbounded params.
                    
                    # Let's stick to a simpler "polish": 
                    # Run a few iterations of perturb with smaller scale.
                    pass
        
        # Cool down
        if temp > min_temp:
            temp *= cooling_rate
        else:
            # Random restart of temperature to escape local minima
            if random.random() < 0.05:
                temp = 0.5
                # And maybe jump to a random point near best
                if random.random() < 0.5:
                    current_support = best_support + np.random.normal(0, 0.1, size=len(best_support))
                    current_support = np.maximum(current_support, 0.0)
                    current_weights = best_weights + np.random.normal(0, 0.1, size=len(best_weights))
                    current_weights = np.maximum(current_weights, 0.0)
                    ws = np.sum(current_weights)
                    if ws > 1e-9:
                        current_weights /= ws
                    current_prob = _calc_prob_fast(current_support, current_weights)

    # Final cleanup
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    else:
        # Ensure validity
        if not _validate(best_support, best_weights):
            best_support = np.array([0.0, 1.0], dtype=np.float64)
            best_weights = np.array([0.5, 0.5], dtype=np.float64)
            
        # Normalize weights
        w_sum = np.sum(best_weights)
        if w_sum > 1e-9:
            best_weights = best_weights / w_sum
        else:
            best_weights = np.ones(len(best_support)) / len(best_support)
            
        # Ensure support >= 0
        best_support = np.maximum(best_support, 0.0)

    return best_support.astype(np.float64), best_weights.astype(np.float64)

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print("Support:", s)
    print("Weights:", w)
    print("Probability:", _calc_prob_fast(s, w))
