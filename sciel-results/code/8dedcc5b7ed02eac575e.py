import numpy as np
import time
import random
from scipy.optimize import differential_evolution

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

def _eval_measure_from_params(params, n):
    """
    Convert parameters to support/weights and evaluate.
    params layout:
    [log_supports (n), log_weights (n)]
    """
    log_s = params[:n]
    log_w = params[n:]
    
    s = np.exp(log_s)
    w = np.exp(log_w)
    
    if not _validate(s, w):
        return 0.0
        
    prob = _calc_prob_fast(s, w)
    return prob

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using global differential evolution search over parameter space.
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
    
    # Use differential evolution for global search over fixed-size support
    # We'll try sizes 2 to 6.
    
    # Bounds for log_supports and log_weights
    # Supports: 0 to 100 -> log: -20 to 5
    # Weights: 1e-6 to 100 -> log: -15 to 5
    
    log_s_bounds = [(-20.0, 5.0)]
    log_w_bounds = [(-15.0, 5.0)]
    
    # Try different sizes
    for n in [2, 3, 4, 5, 6]:
        if time.time() > deadline - 30:
            break
            
        bounds = log_s_bounds * n + log_w_bounds * n
        dim = 2 * n
        
        # Differential evolution settings
        # maxiter and popsize tuned for time budget
        # We have ~500s left, divide among n values.
        # Let's give each n about 80s.
        
        # Objective function to minimize (negate prob)
        def objective(params):
            # Check deadline inside objective? DE doesn't support callbacks easily.
            # We rely on maxiter/popsize to limit.
            return -_eval_measure_from_params(params, n)
        
        try:
            # Use DE
            # popsize=15, maxiter=100 -> 1500 evals.
            # For n=6, dim=12. 1500 evals might be enough.
            # Each eval is fast.
            
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                tol=1e-6, 
                seed=42 + n,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False  # Disable polish to avoid extra time
            )
            
            if res.success and res.fun < 0:
                prob = -res.fun
                if prob > best_prob:
                    params = res.x
                    log_s = params[:n]
                    log_w = params[n:]
                    s = np.exp(log_s)
                    w = np.exp(log_w)
                    
                    if _validate(s, w):
                        best_prob = prob
                        best_support = s.copy()
                        best_weights = w.copy()
                        
        except Exception as e:
            pass # Continue to next n
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # If DE failed or didn't improve, fall back to the original local search
    if best_prob < 0.0 or best_support is None:
        # Re-initialize to original local search logic if needed, 
        # but since we started with a valid candidate, best_support is set.
        # The local search loop from the original code can be added here if time permits.
        pass
    
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

BET: {mechanism: "Replaces the slow local simulated annealing with global Differential Evolution over a log-parameter space, allowing the optimizer to explore the non-convex probability landscape more effectively and find higher-probability support structures that local search might miss.", q: 0.65, delta: [0.0, 0.08]}
