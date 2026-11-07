"""
Erdos minimum overlap: minimize max correlation of h with 1-h.

Approach: Smooth Global Optimization using Soft-Max approximation.
We approximate the max correlation with a log-sum-exp (soft-max) function,
which is smooth and differentiable. We then use gradient-based optimization
(L-BFGS-B via scipy) to find a global minimum, annealing the sharpness
parameter beta. This allows the optimizer to navigate the landscape more
smoothly than discrete local swaps, potentially escaping shallow local minima
trapped by the hard max function.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _value(h):
    """Compute the evaluator's value: max correlation / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return float(np.max(corr)) / N * 2.0

def _project(h):
    """Project h to feasible set: 0<=h<=1, sum(h)=N/2 within tolerance."""
    h = np.clip(h.astype(float), 0.0, 1.0)
    target = N / 2.0
    # Simple iterative projection for sum constraint
    for _ in range(100):
        diff = np.sum(h) - target
        if abs(diff) < 1e-12:
            break
        # Adjust all elements proportionally or uniformly? Uniform is simpler but might hit bounds.
        # We use a water-filling approach: adjust elements not at bounds.
        if diff > 0:
            mask = h > 1e-15
            cnt = np.sum(mask)
            if cnt == 0: break
            h[mask] -= diff / cnt
        else:
            mask = h < 1.0 - 1e-15
            cnt = np.sum(mask)
            if cnt == 0: break
            h[mask] -= diff / cnt
        h = np.clip(h, 0.0, 1.0)
    # Final tiny correction
    diff = np.sum(h) - target
    if abs(diff) > 1e-12:
        idx = np.argmax(h) if diff > 0 else np.argmin(h)
        h[idx] -= diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _soft_max_objective(h, beta):
    """
    Compute the soft-max approximation of the max correlation.
    Objective: (2/N) * (1/beta) * log(sum(exp(beta * corr_k)))
    Also returns the gradient for potential use, though L-BFGS-B uses numerical or analytical.
    Here we provide the function value only, and let scipy compute gradients or we provide them.
    Providing analytical gradients is faster.
    """
    corr = np.correlate(h, 1 - h, mode="full")
    # Scale corr to avoid overflow in exp
    max_corr = np.max(corr)
    # Soft max: log(sum(exp(x))) = max_x + log(sum(exp(x - max_x)))
    exp_terms = np.exp(beta * (corr - max_corr))
    log_sum_exp = max_corr + np.log(np.sum(exp_terms))
    val = (2.0 / N) * log_sum_exp / beta
    return val

def _soft_max_grad(h, beta):
    """
    Analytical gradient of the soft-max objective.
    f(h) = (2/N) * (1/beta) * log(sum_k exp(beta * C_k(h)))
    df/dh = (2/N) * (1/beta) * (1/S) * sum_k exp(beta * C_k) * beta * dC_k/dh
           = (2/N) * (1/S) * sum_k exp(beta * C_k) * dC_k/dh
    where S = sum_k exp(beta * C_k).
    
    C_k(h) = sum_i h_i (1 - h_{i+k})
    dC_k/dh_j = (1 - h_{j+k}) - h_{j-k}  (with boundary handling: if index out of bounds, term is 0)
    Actually, C_k is defined for lags k = -(N-1) .. (N-1).
    Let's compute gradient efficiently.
    """
    corr = np.correlate(h, 1 - h, mode="full")
    max_corr = np.max(corr)
    weights = np.exp(beta * (corr - max_corr))
    S = np.sum(weights)
    
    grad = np.zeros(N)
    # We need sum_k weights[k] * dC_k/dh_j
    # dC_k/dh_j = delta(j in valid range for k) * (1 - h_{j+k}) - delta(j-k in valid range for k) * h_{j-k}
    # This is complex to vectorize directly for all k.
    # Alternative: Numerical gradient is often sufficient for L-BFGS-B if N=100.
    # Let's return None to let scipy use numerical gradient, or implement a simple one.
    # Given N=100, numerical gradient is 100 evaluations. 100*100 = 10k ops per gradient.
    # With ~100 iterations, that's 1M ops. Very fast.
    return None

def _optimize_with_soft_max(start_h, betas, time_limit):
    """
    Run optimization with increasing beta (annealing).
    """
    current_h = start_h.copy()
    best_h = start_h.copy()
    best_val = _value(best_h)
    start_time = time.time()
    
    for beta in betas:
        if time.time() > time_limit:
            break
            
        # Define objective and gradient for scipy
        def obj(x):
            return _soft_max_objective(x, beta)
        
        # Use L-BFGS-B with bounds
        bounds = [(0, 1)] * N
        # Constraint: sum(h) = N/2
        # L-BFGS-B doesn't handle linear equality constraints directly well without penalty.
        # We can add a large penalty for sum deviation in the objective.
        def penalized_obj(x):
            base = _soft_max_objective(x, beta)
            sum_dev = np.sum(x) - N/2.0
            penalty = 1e6 * sum_dev**2
            return base + penalty
        
        def penalized_grad(x):
            # Numerical gradient is default if not provided
            return None

        try:
            res = minimize(penalized_obj, current_h, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6})
            if res.success or res.fun < penalized_obj(current_h):
                candidate = _project(res.x)
                val = _value(candidate)
                if val < best_val:
                    best_val = val
                    best_h = candidate.copy()
                current_h = candidate
        except Exception:
            pass
            
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Seed
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    # Multi-start with different initializations
    # 1. Seed
    # 2. Random
    # 3. Structured (Sine)
    
    initializations = [
        np.full(N, 0.5),
        np.random.default_rng(123).uniform(0, 1, N),
        0.5 + 0.4 * np.sin(2 * np.pi * 3 * np.arange(N) / N),
        0.5 + 0.4 * np.cos(2 * np.pi * 5 * np.arange(N) / N),
        np.random.default_rng(456).uniform(0, 1, N)
    ]
    
    # Betas for annealing: start smooth, end sharp
    betas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    for init in initializations:
        if time.time() > deadline - 10.0:
            break
        init = _project(init)
        h_opt, v_opt = _optimize_with_soft_max(init, betas, deadline - 5.0)
        if v_opt < best_val:
            best_val = v_opt
            best_h = h_opt.copy()
            
    # Final local polish on the best found
    if time.time() < deadline - 5.0:
        # Simple local exchange polish
        rng = np.random.default_rng(999)
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i == j: continue
            # Try small swap
            for delta in [0.01, -0.01, 0.05, -0.05]:
                new_h = best_h.copy()
                new_h[i] += delta
                new_h[j] -= delta
                new_h = _project(new_h)
                v = _value(new_h)
                if v < best_val - 1e-12:
                    best_val = v
                    best_h = new_h
                    break
                    
    return best_h
