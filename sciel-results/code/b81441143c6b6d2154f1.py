# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently using O(n^2 log n) searchsorted.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    if n == 1:
        return 0.0
    
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row
    pair_probs = np.outer(w, w)
    
    sums_flat = pair_sums.flatten()
    probs_flat = pair_probs.flatten()
    
    sort_idx = np.argsort(sums_flat)
    sorted_sums = sums_flat[sort_idx]
    sorted_probs = probs_flat[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    if cum_probs[-1] < 1.0:
        if np.isclose(cum_probs[-1], 1.0, atol=1e-9):
            cum_probs[-1] = 1.0
    
    thresholds = 2.0 * s_row - s_col
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    cdf_values = cum_probs_with_zero[indices]
    
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    return float(P)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game.
    Strategy: Simulated Annealing on variable-length discrete measures.
    Supports adding/removing atoms and perturbing positions/weights.
    """
    t_start = time.time()
    deadline = t_start + 590.0  # Safety margin
    
    # Initial state: A few atoms to allow for structure
    # Start with a geometric distribution which is a reasonable baseline
    n_init = 5
    s = np.geomspace(0.1, 10.0, n_init)
    w = np.ones(n_init) / n_init
    
    best_s = s.copy()
    best_w = w.copy()
    best_P = _compute_P_fast(best_s, best_w)
    
    current_s = s.copy()
    current_w = w.copy()
    current_P = best_P
    
    # Annealing parameters
    T = 1.0
    cooling_rate = 0.995
    min_T = 1e-4
    
    # We will iterate until time runs out or T is too low
    # But we must ensure we have enough time for each evaluation.
    # _compute_P_fast is O(n^2 log n). For n=10, n^2=100, log(100)~6.6. Very fast.
    # For n=50, n^2=2500, log(2500)~7.8. Still fast.
    # Let's cap n at 20 to be safe and keep iterations high.
    max_n = 20
    min_n = 2
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline and T > min_T:
        # Generate a neighbor
        new_s = current_s.copy()
        new_w = current_w.copy()
        n_curr = len(new_s)
        
        # Choose a move: 
        # 1. Perturb support (small)
        # 2. Perturb weights (small)
        # 3. Add an atom
        # 4. Remove an atom
        # 5. Swap/Reorder (implicit in sorting later, but let's keep sorted)
        
        move_type = rng.random()
        
        if move_type < 0.4:
            # Perturb support
            if n_curr > 1:
                idx = rng.integers(0, n_curr)
                # Geometric perturbation
                factor = np.exp(rng.normal(0, 0.1))
                new_s[idx] *= factor
                # Ensure non-negative
                new_s[idx] = max(0.0, new_s[idx])
                
        elif move_type < 0.7:
            # Perturb weights
            if n_curr > 1:
                idx = rng.integers(0, n_curr)
                # Log-uniform perturbation for weights
                # w_i = exp(log_w_i)
                log_w = np.log(new_w + 1e-15)
                log_w[idx] += rng.normal(0, 0.2)
                new_w = np.exp(log_w)
                # Normalize
                s_sum = np.sum(new_w)
                if s_sum > 1e-9:
                    new_w = new_w / s_sum
                else:
                    new_w = np.ones(n_curr) / n_curr
                    
        elif move_type < 0.85 and n_curr < max_n:
            # Add an atom
            # Insert at a random position between existing atoms or near an existing one
            if n_curr == 1:
                # Duplicate and perturb
                new_s = np.concatenate([new_s, [new_s[0] * np.exp(rng.normal(0, 0.5))]])
                new_w = np.concatenate([new_w, [new_w[0]]])
            else:
                # Pick a random index to insert after
                idx = rng.integers(0, n_curr)
                # New support point
                if idx == 0:
                    s_new = new_s[0] * np.exp(rng.normal(0, 0.5))
                elif idx == n_curr - 1:
                    s_new = new_s[-1] * np.exp(rng.normal(0, 0.5))
                else:
                    s_low = new_s[idx]
                    s_high = new_s[idx+1]
                    # Geometric interpolation
                    log_s_low = np.log(max(s_low, 1e-9))
                    log_s_high = np.log(max(s_high, 1e-9))
                    log_s_new = rng.uniform(log_s_low, log_s_high)
                    s_new = np.exp(log_s_new)
                
                # New weight: small fraction of total mass
                # Let's take a small portion from the neighbor
                w_new = new_w[idx] * 0.1
                # Reduce neighbor's weight slightly
                new_w[idx] -= w_new * 0.5 # Don't steal all mass
                
                # Insert
                insert_idx = idx + 1
                new_s = np.insert(new_s, insert_idx, s_new)
                new_w = np.insert(new_w, insert_idx, w_new)
                
                # Re-normalize to ensure sum is 1 (or close)
                s_sum = np.sum(new_w)
                if s_sum > 1e-9:
                    new_w = new_w / s_sum

        elif n_curr > min_n:
            # Remove an atom
            idx = rng.integers(0, n_curr)
            # Redistribute weight to neighbor
            if idx == 0:
                new_w[1] += new_w[0]
            elif idx == n_curr - 1:
                new_w[-2] += new_w[-1]
            else:
                # Split weight to both neighbors
                new_w[idx-1] += new_w[idx] * 0.5
                new_w[idx+1] += new_w[idx] * 0.5
                
            new_s = np.delete(new_s, idx)
            new_w = np.delete(new_w, idx)
            
            # Re-normalize
            s_sum = np.sum(new_w)
            if s_sum > 1e-9:
                new_w = new_w / s_sum
        
        # Validate
        if not (np.all(new_s >= -1e-9) and np.all(new_w >= -1e-9)):
            continue
        if not (np.all(np.isfinite(new_s)) and np.all(np.isfinite(new_w))):
            continue
        if np.sum(new_w) <= 1e-9:
            continue
            
        # Sort by support to keep consistent (though P is invariant to permutation)
        sort_idx = np.argsort(new_s)
        new_s = new_s[sort_idx]
        new_w = new_w[sort_idx]
        
        # Compute new P
        new_P = _compute_P_fast(new_s, new_w)
        
        if not np.isfinite(new_P):
            continue
            
        delta_P = new_P - current_P
        if delta_P > 0 or rng.random() < np.exp(delta_P / T):
            current_s = new_s
            current_w = new_w
            current_P = new_P
            
            if current_P > best_P:
                best_P = current_P
                best_s = current_s.copy()
                best_w = current_w.copy()
                
        T *= cooling_rate
        
        # If T gets very small, we might be stuck. 
        # Occasionally restart T if we haven't improved in a long time?
        # For now, simple cooling.
        
    # Final validation and formatting
    best_s = np.asarray(best_s, dtype=np.float64).flatten()
    best_w = np.asarray(best_w, dtype=np.float64).flatten()
    
    if len(best_s) == 0 or len(best_w) == 0:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    if len(best_s) != len(best_w):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    best_s = np.maximum(best_s, 0.0)
    best_w = np.maximum(best_w, 0.0)
    
    w_sum = np.sum(best_w)
    if w_sum <= 1e-9:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    best_w = best_w / w_sum
    
    if not (np.all(np.isfinite(best_s)) and np.all(np.isfinite(best_w))):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    sort_idx = np.argsort(best_s)
    best_s = best_s[sort_idx]
    best_w = best_w[sort_idx]
    
    return best_s, best_w

# EVOLVE-BLOCK-END
