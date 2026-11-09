# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a small number of support points (n=4) and optimize both support
    positions and weights using scipy.optimize.minimize with multiple starts and polish.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized;
            the evaluator normalizes them to sum to 1).
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin before 630s kill
    
    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] for given support and weights (vectorized)."""
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            return 0.0
        w = weights / w_sum
        
        # Pairwise sums: pair_sums[i,j] = support[i] + support[j]
        pair_sums = support[:, np.newaxis] + support[np.newaxis, :]
        
        # For each (k, l): threshold = 2*s_l - s_k
        # inner(k,l) = sum_{i,j} w_i w_j * 1[pair_sums[i,j] < threshold]
        # P = sum_{k,l} w_k w_l * inner(k,l)
        
        # thresholds[k,l] = 2*support[l] - support[k]
        thresholds = 2.0 * support[np.newaxis, :] - support[:, np.newaxis]  # shape (n, n), [k,l]
        
        # For each (k,l), inner = sum over (i,j) of w_i*w_j where pair_sums[i,j] < thresholds[k,l]
        # pair_sums shape (n,n), thresholds shape (n,n)
        # We need: for each (k,l), sum_{i,j: pair_sums[i,j] < thresholds[k,l]} w_i * w_j
        
        # Efficient approach: sort pair_sums, use cumulative sums
        # But n is small (4), so just do it directly with broadcasting
        # mask[k,l,i,j] = 1[pair_sums[i,j] < thresholds[k,l]]
        # This is n^4 = 256 for n=4, very fast
        
        # inner[k,l] = sum_{i,j} w_i * w_j * 1[pair_sums[i,j] < thresholds[k,l]]
        # = sum_{i,j} w_i * w_j * (pair_sums[i,j] < thresholds[k,l])
        
        # Vectorize: outer product of w with itself gives W[i,j] = w_i * w_j
        W = w[:, np.newaxis] * w[np.newaxis, :]  # shape (n, n)
        
        # For each (k,l): inner[k,l] = sum_{i,j} W[i,j] * 1[pair_sums[i,j] < thresholds[k,l]]
        # Use broadcasting: pair_sums[:, :, np.newaxis, np.newaxis] < thresholds[np.newaxis, np.newaxis, :, :]
        # That gives shape (n, n, n, n) which is 4^4 = 256, fine
        
        # Actually let's do it more efficiently:
        # inner[k,l] = np.sum(W * (pair_sums < thresholds[k,l]))
        # We can vectorize over k,l using einsum or direct sum
        
        # Create 4D array: mask[k,l,i,j] = 1[pair_sums[i,j] < thresholds[k,l]]
        ps = pair_sums[:, :, np.newaxis, np.newaxis]  # (n, n, 1, 1)
        th = thresholds[np.newaxis, np.newaxis, :, :]  # (1, 1, n, n)
        mask = (ps < th)  # (n, n, n, n)
        
        # inner[k,l] = sum_{i,j} W[i,j] * mask[k,l,i,j]
        # = np.einsum('ij,klji->kl', W, mask)
        inner = np.einsum('ij,klji->kl', W, mask)
        
        P = np.sum(w[:, np.newaxis] * w[np.newaxis, :] * inner)
        
        return P
    
    def neg_P(params):
        """Negative of P for minimization. params = [s0, s1, s2, s3, w0, w1, w2, w3]"""
        support = params[:4]
        weights = params[4:]
        
        # Ensure support is non-negative
        support = np.maximum(support, 0.0)
        # Ensure weights are non-negative
        weights = np.maximum(weights, 0.0)
        
        # If all weights are zero, return large penalty
        if np.sum(weights) < 1e-12:
            return 1e10
        
        return -compute_P(support, weights)
    
    # Multiple diverse starting points
    inits = [
        np.array([0.0, 1.0, 2.0, 3.0, 0.25, 0.25, 0.25, 0.25]),
        np.array([0.0, 10.0, 20.0, 30.0, 0.7, 0.1, 0.1, 0.1]),
        np.array([0.0, 1.0, 3.0, 9.0, 0.5, 0.3, 0.15, 0.05]),
        np.array([0.0, 0.5, 1.0, 4.0, 0.4, 0.3, 0.2, 0.1]),
        np.array([0.0, 2.0, 5.0, 10.0, 0.5, 0.25, 0.15, 0.1]),
        np.array([0.0, 1.0, 4.0, 16.0, 0.5, 0.25, 0.15, 0.1]),
        np.array([0.0, 0.1, 1.0, 5.0, 0.6, 0.2, 0.1, 0.1]),
        np.array([0.0, 1.0, 2.0, 8.0, 0.4, 0.3, 0.2, 0.1]),
    ]
    
    best_params = None
    best_P = -1.0
    
    # Phase 1: Nelder-Mead from multiple starts
    for init in inits:
        if time.time() > deadline - 120:
            break
        
        try:
            res = minimize(
                neg_P,
                init,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if res.fun < best_P:
                params = res.x
                support = np.maximum(params[:4], 0.0)
                weights = np.maximum(params[4:], 0.0)
                
                if np.sum(weights) > 1e-12:
                    P_val = -res.fun
                    if P_val > best_P:
                        best_P = P_val
                        best_params = params.copy()
        except Exception:
            pass
    
    # Phase 2: SLSQP polish from best solution
    if best_params is not None and time.time() < deadline - 30:
        try:
            bounds = [(0, None)] * 4 + [(1e-10, 1.0)] * 4
            res2 = minimize(
                neg_P,
                best_params,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'eps': 1e-10}
            )
            
            params2 = res2.x
            support2 = np.maximum(params2[:4], 0.0)
            weights2 = np.maximum(params2[4:], 0.0)
            
            if np.sum(weights2) > 1e-12:
                P_val2 = -res2.fun
                if P_val2 > best_P:
                    best_P = P_val2
                    best_params = params2.copy()
        except Exception:
            pass
    
    # Phase 3: Try COBYLA from best for additional polish
    if best_params is not None and time.time() < deadline - 20:
        try:
            res3 = minimize(
                neg_P,
                best_params,
                method='COBYLA',
                options={'maxiter': 5000, 'rhobeg': 1e-6, 'tol': 1e-14}
            )
            
            params3 = res3.x
            support3 = np.maximum(params3[:4], 0.0)
            weights3 = np.maximum(params3[4:], 0.0)
            
            if np.sum(weights3) > 1e-12:
                P_val3 = -res3.fun
                if P_val3 > best_P:
                    best_P = P_val3
                    best_params = params3.copy()
        except Exception:
            pass
    
    # Final result
    if best_params is not None:
        support = np.maximum(best_params[:4], 0.0)
        weights = np.maximum(best_params[4:], 0.0)
        
        # Ensure validity
        if np.sum(weights) < 1e-9:
            weights = np.ones(4) / 4
            support = np.arange(4, dtype=np.float64)
        
        # Ensure no NaN or inf
        if np.any(np.isnan(support)) or np.any(np.isinf(support)):
            support = np.arange(4, dtype=np.float64)
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            weights = np.ones(4) / 4
        
        return support.astype(np.float64), weights.astype(np.float64)
    else:
        # Fallback to baseline
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights


# EVOLVE-BLOCK-END
