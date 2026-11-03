import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Initial best: uniform
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Strategy: Differential Evolution on a continuous representation of the support.
    # We optimize over 2*N variables:
    #   x[0:N]   -> continuous indices in [0, N-1]
    #   x[N:2N]  -> weights in [0, 1]
    #
    # For each candidate:
    #   1. Round indices to nearest integer.
    #   2. If multiple indices map to the same integer, sum their weights.
    #   3. Construct h of length N.
    #   4. Score h.
    #
    # This allows the optimizer to "discover" the support set dynamically.
    
    def _score_from_de(x):
        # x is length 2N
        indices = np.round(x[:N]).astype(int)
        indices = np.clip(indices, 0, N-1)
        weights = x[N:]
        
        # Ensure non-negative weights
        weights = np.maximum(0, weights)
        
        # If all weights are zero, return 0
        if np.sum(weights) == 0:
            return 0.0
        
        # Aggregate weights for same indices
        h = np.zeros(N)
        np.add.at(h, indices, weights)
        
        # Normalize to L1=1 for stability during search (ratio is scale invariant)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        h = h / l1
        
        return _score_raw(h)
    
    # Bounds for DE
    # Indices: [0, N-1]
    # Weights: [0, 1]
    bounds = [(0.0, N-1.0)] * N + [(0.0, 1.0)] * N
    
    # Run Differential Evolution
    # Use a moderate population size and max iterations to balance exploration/exploitation
    # popsize=20, maxiter=100 -> 2000 evaluations. Each eval is ~1ms. Total ~2s.
    # We can afford more. Let's try popsize=30, maxiter=200 -> 6000 evals. ~6s.
    
    try:
        de_res = differential_evolution(
            lambda x: -_score_from_de(x),  # Minimize negative score
            bounds,
            maxiter=200,
            popsize=30,
            tol=1e-6,
            seed=42,
            polish=False,  # We'll do our own polish
            init='sobol'  # Better initial distribution
        )
        
        if de_res.success or de_res.fun < 0:
            x_best_de = de_res.x
            s_de = -de_res.fun
            if s_de > best_score:
                best_score = s_de
                # Reconstruct h for best
                indices = np.round(x_best_de[:N]).astype(int)
                indices = np.clip(indices, 0, N-1)
                weights = np.maximum(0, x_best_de[N:])
                h_candidate = np.zeros(N)
                np.add.at(h_candidate, indices, weights)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                    best_h = h_candidate.copy()
                    
    except Exception as e:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        best_h = np.maximum(0, best_h)
        if np.sum(best_h) == 0:
            best_h = np.ones(N) / N
        else:
            best_h /= np.sum(best_h)
        return list(best_h)
    
    # Polish with SLSQP on the continuous weights, keeping the support fixed from DE
    # Extract support from best_h
    support_mask = best_h > 1e-8
    support_indices = np.where(support_mask)[0]
    
    if len(support_indices) > 0:
        # Define objective for SLSQP on just the support weights
        def neg_obj_slsqp(w):
            h = np.zeros(N)
            h[support_indices] = w
            l1 = np.sum(h)
            if l1 == 0:
                return 0.0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0:
                return 0.0
            return -(l2_sq / (l1 * max_conv))
        
        # Initial weights from best_h
        w_init = best_h[support_indices]
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds_slsqp = [(0.0, 1.0)] * len(support_indices)
        
        try:
            res = minimize(
                neg_obj_slsqp,
                w_init,
                method='SLSQP',
                bounds=bounds_slsqp,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res.success:
                h_candidate = np.zeros(N)
                h_candidate[support_indices] = np.maximum(0, res.x)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
        except Exception:
            pass
    
    # Final cleanup
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) == 0:
        best_h = np.ones(N) / N
    else:
        best_h /= np.sum(best_h)
    
    return list(best_h)
