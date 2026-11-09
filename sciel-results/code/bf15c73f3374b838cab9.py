# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a small number of support points (n=4) and optimize both support
    positions and weights using scipy.optimize.minimize with SLSQP.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized;
            the evaluator normalizes them to sum to 1).
    """
    start_time = time.time()
    deadline = start_time + 570  # 570s safety margin before 630s kill
    
    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] for given support and weights."""
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            return 0.0
        w = weights / w_sum
        
        # Compute all pairwise sums s_i + s_j
        s_i = support[:, np.newaxis]
        s_j = support[np.newaxis, :]
        pair_sums = s_i + s_j  # shape (n, n)
        
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[l] - support[k]
                mask = pair_sums < threshold
                inner = np.sum(w[np.newaxis, :] * w[:, np.newaxis] * mask)
                P += w[k] * w[l] * inner
        
        return P
    
    def neg_P(params):
        """Negative of P for minimization. params = [s0, s1, s2, s3, w0, w1, w2, w3]"""
        support = np.maximum(params[:4], 0.0)
        weights = np.maximum(params[4:], 0.0)
        
        if np.sum(weights) < 1e-12:
            return 1e10
        
        return -compute_P(support, weights)
    
    # Initial guesses: various starting points
    inits = [
        # Uniform on {0, 1, 2, 3}
        np.array([0.0, 1.0, 2.0, 3.0, 0.25, 0.25, 0.25, 0.25]),
        # Concentrated at 0
        np.array([0.0, 5.0, 10.0, 20.0, 0.6, 0.2, 0.1, 0.1]),
        # Geometric-like
        np.array([0.0, 1.0, 3.0, 9.0, 0.5, 0.3, 0.15, 0.05]),
        # Spread out
        np.array([0.0, 2.0, 6.0, 18.0, 0.4, 0.3, 0.2, 0.1]),
        # Two-point-like
        np.array([0.0, 1.0, 100.0, 200.0, 0.8, 0.1, 0.05, 0.05]),
        # Another spread
        np.array([0.0, 0.5, 2.0, 8.0, 0.3, 0.3, 0.25, 0.15]),
    ]
    
    best_params = None
    best_P = -1.0
    
    bounds = [(0, None)] * 4 + [(0, 1)] * 4
    
    for init in inits:
        if time.time() > deadline - 60:
            break
        
        try:
            # Use SLSQP for better handling of piecewise smooth objective
            res = minimize(
                neg_P,
                init,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
            )
            
            if res.fun < best_P or best_params is None:
                params = res.x
                support = np.maximum(params[:4], 0.0)
                weights = np.maximum(params[4:], 0.0)
                
                if np.sum(weights) > 1e-12:
                    P_val = -res.fun
                    if P_val > best_P:
                        best_P = P_val
                        best_params = params.copy()
        except Exception as e:
            pass
    
    # If we found a good solution, try a polish pass with Nelder-Mead for fine-tuning
    if best_params is not None and time.time() < deadline - 30:
        try:
            res_polish = minimize(
                neg_P,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_polish.fun < best_P:
                params = res_polish.x
                support = np.maximum(params[:4], 0.0)
                weights = np.maximum(params[4:], 0.0)
                
                if np.sum(weights) > 1e-12:
                    P_val = -res_polish.fun
                    if P_val > best_P:
                        best_P = P_val
                        best_params = params.copy()
        except Exception as e:
            pass
    
    # If we found a good solution, use it
    if best_params is not None:
        support = np.maximum(best_params[:4], 0.0)
        weights = np.maximum(best_params[4:], 0.0)
        
        # Ensure validity
        if np.sum(weights) < 1e-9:
            weights = np.ones(4) / 4
            support = np.arange(4, dtype=np.float64)
        
        return support.astype(np.float64), weights.astype(np.float64)
    else:
        # Fallback to baseline
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights


# EVOLVE-BLOCK-END
