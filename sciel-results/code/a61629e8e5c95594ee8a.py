"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _exact_c3(h):
    """Compute exact C3 upper bound: max|conv| / (sum|h|)^2"""
    conv = np.convolve(h, h, mode='full')
    max_abs = np.max(np.abs(conv))
    sum_abs = np.sum(np.abs(h))
    if sum_abs < 1e-12:
        return np.inf
    return max_abs / (sum_abs ** 2)


def _discrete_local_search(h, max_rounds=30, time_limit=None):
    """
    Local search by flipping individual signs to reduce C3.
    h is a signed step function (values near +/-1 after tanh).
    """
    h = h.copy()
    current_c3 = _exact_c3(h)
    
    for round_idx in range(max_rounds):
        if time_limit is not None and time.time() > time_limit:
            break
        
        improved = False
        for i in range(N):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Try flipping sign of h[i]
            h_old = h[i]
            h[i] = -h[i]
            new_c3 = _exact_c3(h)
            
            if new_c3 < current_c3 - 1e-12:
                current_c3 = new_c3
                improved = True
            else:
                h[i] = h_old
        
        if not improved:
            break
    
    return h


def _continuous_optimize(x0, deadline, maxiter=200):
    """Run L-BFGS-B on tanh-parameterized objective."""
    def objective(x):
        h = np.tanh(x)
        return _exact_c3(h)
    
    res = minimize(objective, x0, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
    return res.x, res.fun


def construct():
    np.random.seed(42)
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = None
    best_val = np.inf
    
    # Generate diverse structured initializations
    initializations = []
    
    # 1. All positive (large x)
    initializations.append(np.full(N, 5.0))
    
    # 2. Alternating signs
    alt = np.zeros(N)
    for i in range(N):
        alt[i] = 5.0 if i % 2 == 0 else -5.0
    initializations.append(alt)
    
    # 3. Block pattern: +++++-----+++++-----... (period 10)
    block = np.zeros(N)
    for i in range(N):
        block[i] = 5.0 if (i // 5) % 2 == 0 else -5.0
    initializations.append(block)
    
    # 4. Block pattern: period 8
    block8 = np.zeros(N)
    for i in range(N):
        block8[i] = 5.0 if (i // 4) % 2 == 0 else -5.0
    initializations.append(block8)
    
    # 5. Random
    initializations.append(np.random.normal(0, 2, N))
    
    # 6. Another random
    initializations.append(np.random.normal(0, 1, N))
    
    # 7. Ramp pattern
    ramp = np.linspace(-5, 5, N)
    initializations.append(ramp)
    
    # 8. Two blocks: first half +, second half -
    two_block = np.zeros(N)
    two_block[:N//2] = 5.0
    two_block[N//2:] = -5.0
    initializations.append(two_block)
    
    for idx, x0 in enumerate(initializations):
        if time.time() > deadline - 5:
            break
        
        # Continuous optimization
        x_opt, val = _continuous_optimize(x0.copy(), deadline, maxiter=150)
        h_opt = np.tanh(x_opt)
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
        
        # Discrete local search refinement
        if time.time() < deadline - 10:
            h_refined = _discrete_local_search(h_opt, max_rounds=20, time_limit=deadline - 5)
            val_refined = _exact_c3(h_refined)
            if val_refined < best_val:
                best_val = val_refined
                best_h = h_refined.copy()
    
    # Final polish: if we have a good solution, do more discrete search
    if best_h is not None and time.time() < deadline - 5:
        h_polished = _discrete_local_search(best_h, max_rounds=50, time_limit=deadline - 2)
        val_polished = _exact_c3(h_polished)
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished
    
    if best_h is None:
        # Fallback: return alternating signs
        best_h = np.zeros(N)
        for i in range(N):
            best_h[i] = 1.0 if i % 2 == 0 else -1.0
    
    return best_h
# EVOLVE-BLOCK-END
