import numpy as np
import time
from scipy.optimize import minimize

def _calc_prob_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for a discrete measure.
    Uses vectorized operations for speed.
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
    
    # Pre-sort S_flat and corresponding W_flat for prefix sum queries
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    W_cumsum = np.cumsum(W_sorted)
    W_cumsum_padded = np.concatenate(([0.0], W_cumsum))
    
    for k in range(n):
        # We need P[X1 + X2 + X3 < 2*s_k]
        # This is sum_{l} p_l * P[X1 + X2 < 2*s_k - s_l]
        # For fixed l, P[X1 + X2 < val] is sum_{i,j} p_i p_j I(s_i+s_j < val)
        # Which is W_cumsum_padded[searchsorted(S_sorted, val, side='left')]
        
        thresholds_l = 2.0 * support[k] - support
        indices = np.searchsorted(S_sorted, thresholds_l, side='left')
        sums_for_l = W_cumsum_padded[indices]
        
        prob3 = np.dot(p, sums_for_l)
        total_prob += p[k] * prob3
        
    return total_prob

def _eval_measure(support, weights):
    """
    Evaluate the probability for a given support and weights.
    Returns -P for minimization (so lower is better for minimizers).
    """
    if len(support) != len(weights):
        return 1.0
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return 1.0
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return 1.0
    if np.sum(weights) <= 1e-9:
        return 1.0
    
    # Ensure strictly increasing support for uniqueness (optional but good for SA)
    # Actually, the problem doesn't require strictly increasing support, just distinct points?
    # "discrete measure mu: point s_i has probability w_i"
    # If s_i == s_j, they are effectively the same point.
    # The metric sums over k, l, i, j.
    # If s_i == s_j, it's just a higher weight at that point.
    # So we should allow non-strictly increasing?
    # However, for the SA parameterization, it's easier to keep them sorted.
    # Let's check if sorting changes the value.
    # The value depends on the multiset of (s, w).
    # So we can sort support and permute weights accordingly.
    
    # Let's just compute directly.
    p = _calc_prob_fast(support, weights)
    return -p

def _sa_step(support, weights, temp, rng):
    """
    Perform a single simulated annealing step.
    Returns new_support, new_weights.
    """
    n = len(support)
    new_support = support.copy()
    new_weights = weights.copy()
    
    # Randomly choose a move type
    # 1. Perturb support (scale)
    # 2. Perturb weights (Dirichlet-like)
    # 3. Add a point
    # 4. Remove a point
    
    move_type = rng.randint(0, 3)
    
    if move_type == 0:
        # Perturb support
        # Scale each support point slightly
        # Use log-space perturbation for positive values
        # support must be >= 0.
        # Let's perturb in log space for non-zero, or additive for zero.
        # To keep it simple, let's just do multiplicative perturbation for s > 0,
        # and additive for s == 0 (though s=0 is often fixed or rare).
        
        # Better: perturb the vector of log-supports (with 0 mapped to -inf? No, s>=0).
        # Let's use a transformation: s_i = exp(x_i) for i>0, s_0=0?
        # Or just perturb s_i directly with reflection at 0.
        
        # Let's perturb the log of (s_i + eps)
        eps = 1e-6
        log_s = np.log(new_support + eps)
        log_s += rng.normal(0, 0.1 * temp, size=n)
        new_support = np.exp(log_s) - eps
        new_support = np.maximum(new_support, 0.0)
        
    elif move_type == 1:
        # Perturb weights
        # Add noise to weights and renormalize
        # Ensure weights stay positive
        noise = rng.normal(0, 0.05 * temp, size=n)
        new_weights = new_weights + noise
        new_weights = np.maximum(new_weights, 0.0)
        w_sum = np.sum(new_weights)
        if w_sum > 0:
            new_weights = new_weights / w_sum
        else:
            # Fallback to uniform
            new_weights = np.ones(n) / n
            
    elif move_type == 2 and n < 8:
        # Add a point
        # Insert a new point with small weight
        # Choose a random position to insert
        # To maintain sorted order, we can just append and sort, or insert.
        # Let's append and sort.
        new_s = rng.uniform(0.1, 10.0)
        new_w = 0.01
        # Reduce existing weights to make room?
        # Or just add and renormalize.
        new_support = np.append(new_support, new_s)
        new_weights = np.append(new_weights, new_w)
        # Sort
        sort_idx = np.argsort(new_support)
        new_support = new_support[sort_idx]
        new_weights = new_weights[sort_idx]
        # Renormalize weights? The measure definition normalizes by sum(weights).
        # So we don't need to renormalize for validity, but for optimization stability,
        # it's good to keep sum(weights) ~ 1.
        new_weights = new_weights / np.sum(new_weights)
        
    elif move_type == 3 and n > 2:
        # Remove a point
        # Remove a random point (not the first or last to avoid edge issues? No, any)
        # But we must keep n >= 1.
        idx_to_remove = rng.randint(0, n-1)
        new_support = np.delete(new_support, idx_to_remove)
        new_weights = np.delete(new_weights, idx_to_remove)
        if len(new_weights) > 0:
            new_weights = new_weights / np.sum(new_weights)
            
    return new_support, new_weights

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using Simulated Annealing with variable support size.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_prob = -1.0
    best_support = None
    best_weights = None
    
    rng = np.random.default_rng(42)
    
    # Initial state: 3 points
    current_support = np.array([0.0, 1.0, 5.0])
    current_weights = np.array([0.5, 0.3, 0.2])
    current_prob = _calc_prob_fast(current_support, current_weights)
    
    if current_prob > best_prob:
        best_prob = current_prob
        best_support = current_support.copy()
        best_weights = current_weights.copy()
        
    # SA Parameters
    T_start = 1.0
    T_end = 0.001
    cooling_rate = 0.999
    
    T = T_start
    step_count = 0
    
    while T > T_end and time.time() < deadline - 10:
        # Perform a move
        new_support, new_weights = _sa_step(current_support, current_weights, T, rng)
        
        # Evaluate
        new_prob = _calc_prob_fast(new_support, new_weights)
        
        # Check validity (basic)
        if np.any(np.isnan(new_support)) or np.any(np.isnan(new_weights)) or \
           np.any(new_support < -1e-9) or np.any(new_weights < -1e-9) or \
           np.sum(new_weights) <= 1e-9:
            new_prob = -1.0 # Penalty
            
        delta = new_prob - current_prob
        
        # Acceptance criterion
        if delta > 0:
            # Accept if improvement
            accept = True
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            if rng.random() < np.exp(delta / T):
                accept = True
            else:
                accept = False
                
        if accept:
            current_support = new_support
            current_weights = new_weights
            current_prob = new_prob
            
            if current_prob > best_prob:
                best_prob = current_prob
                best_support = current_support.copy()
                best_weights = current_weights.copy()
                
        # Cool down
        T *= cooling_rate
        step_count += 1
        
        # Check time
        if time.time() > deadline - 10:
            break
            
    # Polish phase: Use Nelder-Mead on the best found support/weights
    # We need to convert best_support/weights to a parameter vector for Nelder-Mead.
    # Parameterization: log(s_i) for s_i > 0, and weights.
    # But s_0 might be 0.
    # Let's use a simpler approach: just perturb the best support/weights directly
    # using a local search or Nelder-Mead on the raw values with constraints handled by penalty.
    
    if best_support is not None and time.time() < deadline - 5:
        n_best = len(best_support)
        
        def objective(params):
            s = params[:n_best]
            w = params[n_best:]
            
            # Constraints: s >= 0, w >= 0, sum(w) > 0
            if np.any(s < -1e-9) or np.any(w < -1e-9):
                return 1.0
            if np.sum(w) <= 1e-9:
                return 1.0
                
            p = _calc_prob_fast(s, w)
            return -p
            
        # Initial guess
        init_params = np.concatenate([best_support, best_weights])
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res.fun < 0:
                p = -res.fun
                if p > best_prob:
                    best_prob = p
                    best_support = res.x[:n_best]
                    best_weights = res.x[n_best:]
        except Exception:
            pass
            
    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0, 10.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        best_prob = _calc_prob_fast(best_support, best_weights)

    # Final cleanup
    if np.any(best_weights < -1e-9) or np.sum(best_weights) <= 1e-9 or np.any(best_support < -1e-9):
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    
    wsum = np.sum(best_weights)
    if wsum > 1e-9:
        best_weights = best_weights / wsum
        
    best_support = np.maximum(best_support, 0.0)
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print("Support:", s)
    print("Weights:", w)
    print("Probability:", _calc_prob_fast(s, w))
