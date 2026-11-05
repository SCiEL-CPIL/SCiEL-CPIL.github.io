"""QAOA parameter design with gradient-based optimization and adaptive multi-start search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def _precompute_cut_values(n, edges):
    """
    Precompute the cut value for every possible state mask.
    Returns an array of shape (2^n,).
    """
    dim = 1 << n
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        val = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                val += 1
        cut_vals[mask] = val
    return cut_vals

def _apply_mixer_fast(psi, n, beta):
    """
    Apply exp(-i * beta * sum(X_i)) to the state vector psi.
    Uses a fast approach with reshape and axis manipulation.
    """
    cb = np.cos(beta)
    sb = -1j * np.sin(beta)
    
    shape = (2,) * n
    psi_t = psi.reshape(shape)
    
    for i in range(n):
        psi_moved = np.moveaxis(psi_t, i, 0)
        half_0 = psi_moved[0].copy()
        half_1 = psi_moved[1].copy()
        psi_moved[0] = cb * half_0 + sb * half_1
        psi_moved[1] = sb * half_0 + cb * half_1
        psi_t = np.moveaxis(psi_moved, 0, i)
    
    return psi_t.reshape(-1)

def _qaoa_state(n, edges, gammas, betas, cut_vals):
    """
    Compute the final state vector after QAOA evolution.
    Returns the state vector of length 2^n.
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    p = len(gammas)
    
    for k in range(p):
        # Cost unitary
        psi *= np.exp(-1j * gammas[k] * cut_vals)
        # Mixer unitary
        psi = _apply_mixer_fast(psi, n, betas[k])
        
    return psi

def _qaoa_expectation(n, edges, gammas, betas, cut_vals):
    """
    Compute the expected cut size.
    """
    psi = _qaoa_state(n, edges, gammas, betas, cut_vals)
    probs = np.abs(psi) ** 2
    return np.dot(probs, cut_vals)

def _qaoa_gradient(n, edges, gammas, betas, cut_vals):
    """
    Compute the gradient of the expected cut with respect to [g0, b0, g1, b1]
    using the parameter-shift rule.
    """
    dim = 1 << n
    shift = np.pi / 2
    grad = np.zeros(4, dtype=np.float64)
    
    # Parameter indices: 0=g0, 1=b0, 2=g1, 3=b1
    for idx in range(4):
        # Shift positive
        g_plus = gammas.copy()
        b_plus = betas.copy()
        if idx < 2:
            g_plus[idx] += shift
        else:
            b_plus[idx - 2] += shift
        exp_plus = _qaoa_expectation(n, edges, g_plus, b_plus, cut_vals)
        
        # Shift negative
        g_minus = gammas.copy()
        b_minus = betas.copy()
        if idx < 2:
            g_minus[idx] -= shift
        else:
            b_minus[idx - 2] -= shift
        exp_minus = _qaoa_expectation(n, edges, g_minus, b_minus, cut_vals)
        
        grad[idx] = (exp_plus - exp_minus) / 2
    
    return grad

def _optimize_angles(n, edges, p, start_time, timeout):
    """
    Optimize angles using L-BFGS-B with gradient-based optimization and multi-start.
    """
    # Precompute cut values
    cut_vals = _precompute_cut_values(n, edges)
    
    # Objective function for minimization (negative expectation)
    def objective(params):
        g0, b0, g1, b1 = params
        gammas = np.array([g0, g1])
        betas = np.array([b0, b1])
        try:
            exp_cut = _qaoa_expectation(n, edges, gammas, betas, cut_vals)
            return -exp_cut
        except Exception:
            return 1e9
    
    def gradient(params):
        g0, b0, g1, b1 = params
        gammas = np.array([g0, g1])
        betas = np.array([b0, b1])
        try:
            grad = _qaoa_gradient(n, edges, gammas, betas, cut_vals)
            return -grad
        except Exception:
            return np.zeros(4)
    
    best_score = -np.inf
    best_params = None
    
    # Measure time for a single evaluation to estimate budget
    t0 = time.time()
    _qaoa_expectation(n, edges, np.array([0.5, 0.5]), np.array([0.5, 0.5]), cut_vals)
    eval_time = time.time() - t0
    
    # Estimate how many L-BFGS-B runs we can afford
    # Each L-BFGS-B run takes ~50-100 function evaluations
    # With gradient, each eval is 1 cost + 8 gradient evals = 9 total
    # So ~500-900 total QAOA evaluations per run
    # If eval_time is 50us, that's ~25-45ms per run
    # In 120s, we can do ~2600-4800 runs. Let's be conservative.
    
    # Adaptive number of restarts
    base_restarts = 50
    if eval_time < 0.001:  # Fast evaluation
        num_restarts = 200
    elif eval_time < 0.005:
        num_restarts = 100
    else:
        num_restarts = 50
    
    np.random.seed(42)
    
    for r in range(num_restarts):
        # Check time
        if time.time() - start_time > timeout - 10:
            break
        
        # Random initialization
        init = np.random.uniform(0, np.pi, size=4)
        
        # L-BFGS-B options
        options = {
            'maxiter': 100,
            'ftol': 1e-8,
            'gtol': 1e-6,
            'maxls': 20
        }
        
        bounds = [(0, np.pi), (0, np.pi), (0, np.pi), (0, np.pi)]
        
        try:
            res = scipy.optimize.minimize(
                objective, init, method='L-BFGS-B', 
                jac=gradient, bounds=bounds, options=options
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            continue
    
    # Final high-precision polish pass on the best parameters found
    if best_params is not None and time.time() - start_time < timeout - 3:
        try:
            polish_options = {
                'maxiter': 300,
                'ftol': 1e-10,
                'gtol': 1e-8,
                'maxls': 30
            }
            res_polish = scipy.optimize.minimize(
                objective, best_params, method='L-BFGS-B',
                jac=gradient, bounds=bounds, options=polish_options
            )
            polish_score = -res_polish.fun
            if polish_score > best_score:
                best_score = polish_score
                best_params = res_polish.x.copy()
        except Exception:
            pass
    
    if best_params is not None:
        # Reshape to (p, 2)
        return np.array([[best_params[0], best_params[1]],
                         [best_params[2], best_params[3]]])
    else:
        # Fallback
        return np.array([[0.35, 0.25], [0.35, 0.25]])

def angles(n, edges, p):
    start_time = time.time()
    timeout = 145  # Safety margin from 150s limit
    
    # Optimize angles
    optimal_angles = _optimize_angles(n, edges, p, start_time, timeout)
    
    # Final check
    if time.time() - start_time > timeout:
        return np.array([[0.35, 0.25]] * p)
        
    return optimal_angles
# EVOLVE-BLOCK-END
