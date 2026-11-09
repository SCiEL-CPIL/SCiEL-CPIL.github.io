# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.optimize import linprog
from scipy.stats import dirichlet
import itertools

def _evaluate_measure_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently using sorted pairwise sums.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    
    # Build all pairwise sums and their weights
    # S_ij = s_i + s_j, weight = w_i * w_j
    S = support[:, None] + support[None, :]  # (n, n)
    W = w[:, None] * w[None, :]  # (n, n)
    
    S_flat = S.flatten()  # (n^2,)
    W_flat = W.flatten()  # (n^2,)
    
    # Sort by S value
    order = np.argsort(S_flat, kind='stable')
    S_sorted = S_flat[order]
    W_sorted = W_flat[order]
    
    # Prefix sums of W_sorted for fast range queries
    W_prefix = np.concatenate([[0.0], np.cumsum(W_sorted)])
    
    # For each (k, l), target = 2*s_l - s_k
    # Need sum of W_flat[i] where S_flat[i] < target
    # Use binary search on S_sorted
    
    T = 2.0 * support[None, :] - support[:, None]  # (n, n), T[k,l] = 2*s_l - s_k
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            target = T[k, l]
            # Find index of first element >= target (strict inequality: < target)
            idx = np.searchsorted(S_sorted, target, side='left')
            prob_kl = W_prefix[idx]
            P += w[k] * w[l] * prob_kl
    
    return P

def _objective(params, n):
    """
    Objective function for Nelder-Mead: minimize -P.
    """
    log_s = params[:n]
    log_w = params[n:]
    
    support = np.exp(log_s)
    max_log_w = np.max(log_w)
    w_unnorm = np.exp(log_w - max_log_w)
    weights = w_unnorm / np.sum(w_unnorm)
    
    P = _evaluate_measure_fast(support, weights)
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Optimize the discrete measure using Nelder-Mead with fast evaluation.
    """
    start_time = time.time()
    deadline = start_time + 585
    
    best_score = -np.inf
    best_support = None
    best_weights = None
    
    random.seed(42)
    np.random.seed(42)
    
    candidate_ns = [3, 4, 5, 6, 7]
    
    for n in candidate_ns:
        num_restarts = 40
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize support and weights
            if n == 3 and r < 15:
                # Known good 3-point patterns
                if r % 3 == 0:
                    support = np.array([0.0, 1.0, 10.0])
                elif r % 3 == 1:
                    support = np.array([0.0, 0.5, 8.0])
                else:
                    support = np.array([0.0, 1.5, 12.0])
            elif r % 4 == 0:
                # Structured start with spread
                support = np.sort(np.random.uniform(0, 1, n - 2))
                support = np.concatenate([support, [np.random.uniform(5, 10), np.random.uniform(10, 20)]])
                support = np.sort(support)
            elif r % 4 == 1:
                # Geometric spacing
                ratio = np.random.uniform(1.5, 3.0)
                support = np.array([0.0])
                for i in range(1, n):
                    support = np.append(support, support[-1] * ratio + np.random.uniform(0, 0.5))
                support = np.sort(support)
            elif r % 4 == 2:
                # Random with one very large
                support = np.sort(np.random.uniform(0, 3, n - 1))
                support = np.concatenate([support, [np.random.uniform(8, 25)]])
            else:
                # Random in [0, 10]
                support = np.sort(np.random.uniform(0, 10, n))
            
            # Weights: various initializations
            if r % 3 == 0:
                weights = np.random.dirichlet(np.ones(n))
            elif r % 3 == 1:
                # Heavier on middle
                weights = np.random.dirichlet(np.ones(n) * 0.5)
                weights[n//2] += 1.0
                weights /= np.sum(weights)
            else:
                # Heavier on large values
                weights = np.random.dirichlet(np.ones(n))
                weights[-1] += 1.0
                weights /= np.sum(weights)
            
            # Ensure support is non-negative
            support = np.maximum(support, 0.0)
            
            # Convert to log-space
            log_s = np.log(support + 1e-15)
            log_w = np.log(weights + 1e-15)
            
            # Initial objective
            current_score = _evaluate_measure_fast(support, weights)
            if current_score > best_score:
                best_score = current_score
                best_support = support.copy()
                best_weights = weights.copy()
            
            # Run Nelder-Mead optimization
            try:
                result = minimize(
                    _objective,
                    x0=np.concatenate([log_s, log_w]),
                    args=(n,),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                
                opt_params = result.x
                opt_log_s = opt_params[:n]
                opt_log_w = opt_params[n:]
                
                opt_support = np.exp(opt_log_s)
                max_log_w = np.max(opt_log_w)
                opt_w_unnorm = np.exp(opt_log_w - max_log_w)
                opt_weights = opt_w_unnorm / np.sum(opt_w_unnorm)
                
                opt_score = _evaluate_measure_fast(opt_support, opt_weights)
                if opt_score > current_score:
                    current_score = opt_score
                    support = opt_support
                    weights = opt_weights
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_support = support.copy()
                        best_weights = weights.copy()
            except Exception:
                pass
            
            if time.time() > deadline:
                break
    
    # If we didn't find anything, fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0, 10.0])
        best_weights = np.array([0.3, 0.4, 0.3])
    
    # Final validity check
    if np.any(np.isnan(best_support)) or np.any(np.isinf(best_support)) or \
       np.any(np.isnan(best_weights)) or np.any(np.isinf(best_weights)) or \
       np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
        best_support = np.array([0.0, 1.0, 10.0])
        best_weights = np.array([0.3, 0.4, 0.3])
    
    # Ensure weights sum to something reasonable
    w_sum = np.sum(best_weights)
    if w_sum <= 1e-9:
        best_weights = np.array([0.3, 0.4, 0.3])
    
    return best_support, best_weights
# EVOLVE-BLOCK-END
