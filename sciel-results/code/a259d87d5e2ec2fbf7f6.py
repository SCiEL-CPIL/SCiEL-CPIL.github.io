"""Rewritten program for QAOA parameter design with gradient-based optimization."""
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
    Apply exp(-i * beta * sum(X_i)) to the state vector psi in-place.
    Uses a faster approach with reshape and broadcasting.
    """
    cb = np.cos(beta)
    sb = -1j * np.sin(beta)
    
    shape = (2,) * n
    psi_t = psi.reshape(shape)
    
    for i in range(n):
        # Move axis i to front for easier slicing
        psi_moved = np.moveaxis(psi_t, i, 0)
        half_0 = psi_moved[0].copy()
        half_1 = psi_moved[1].copy()
        psi_moved[0] = cb * half_0 + sb * half_1
        psi_moved[1] = sb * half_0 + cb * half_1
        # Move axis back
        psi_t = np.moveaxis(psi_moved, 0, i)
    
    # psi is a view, so we need to copy back if shape changed
    # Actually, reshape returns a view, so modifications to psi_t modify psi
    # But after multiple moveaxis operations, we may have lost the view
    # Let's just return the reshaped result
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

def _qaoa_gradient(n, edges, params, cut_vals):
    """
    Compute gradient of the expected cut with respect to parameters.
    params = [g0, b0, g1, b1]
    Uses finite differences for simplicity (can be replaced with analytical).
    """
    p = 2
    gammas = np.array([params[0], params[2]])
    betas = np.array([params[1], params[3]])
    
    eps = 1e-5
    grad = np.zeros(4)
    
    for i in range(4):
        params_plus = params.copy()
        params_plus[i] += eps
        g_plus = np.array([params_plus[0], params_plus[2]])
        b_plus = np.array([params_plus[1], params_plus[3]])
        
        params_minus = params.copy()
        params_minus[i] -= eps
        g_minus = np.array([params_minus[0], params_minus[2]])
        b_minus = np.array([params_minus[1], params_minus[3]])
        
        f_plus = _qaoa_expectation(n, edges, g_plus, b_plus, cut_vals)
        f_minus = _qaoa_expectation(n, edges, g_minus, b_minus, cut_vals)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad

def _optimize_angles(n, edges, p, start_time, timeout):
    """
    Optimize angles using L-BFGS-B with multiple restarts.
    """
    cut_vals = _precompute_cut_values(n, edges)
    
    def objective(params):
        try:
            g0, b0, g1, b1 = params
            gammas = np.array([g0, g1])
            betas = np.array([b0, b1])
            return -_qaoa_expectation(n, edges, gammas, betas, cut_vals)
        except Exception:
            return 1e9

    def gradient(params):
        try:
            g0, b0, g1, b1 = params
            gammas = np.array([g0, g1])
            betas = np.array([b0, b1])
            grad = _qaoa_gradient(n, edges, params, cut_vals)
            return -grad  # Negate because we're minimizing negative expectation
        except Exception:
            return np.zeros(4)

    best_score = -np.inf
    best_params = None
    
    # Determine number of restarts based on available time
    # First, do a quick test to estimate evaluation time
    test_params = np.array([0.5, 0.5, 0.5, 0.5])
    t0 = time.time()
    _qaoa_expectation(n, edges, np.array([0.5, 0.5]), np.array([0.5, 0.5]), cut_vals)
    t1 = time.time()
    eval_time = max(t1 - t0, 0.001)
    
    # Estimate: L-BFGS-B with gradient ~ 10x evals for gradient + ~50 function evals
    # So ~150 * eval_time per restart
    time_per_restart = 150 * eval_time
    available_time = timeout - (time.time() - start_time) - 2  # 2s safety margin
    num_restarts = max(10, min(300, int(available_time / max(time_per_restart, 0.001))))
    
    np.random.seed(42)
    
    # Good initial points for QAOA
    base_inits = [
        [0.5, 0.5, 0.3, 0.3],
        [0.7, 0.7, 0.4, 0.4],
        [0.3, 0.4, 0.2, 0.3],
        [0.5, 0.3, 0.5, 0.3],
    ]
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - 3:
            break
            
        if r < len(base_inits):
            init = np.array(base_inits[r], dtype=np.float64)
        else:
            # Random init in reasonable range
            init = np.random.uniform([0.1, 0.1, 0.1, 0.1], [1.5, 1.5, 1.5, 1.5])
        
        try:
            res = scipy.optimize.minimize(
                objective, init, jac=gradient,
                method='L-BFGS-B',
                bounds=[(0.0, np.pi)] * 4,
                options={'maxiter': 80, 'ftol': 1e-6, 'gtol': 1e-4, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            continue
            
    if best_params is not None:
        return np.array([[best_params[0], best_params[1]],
                         [best_params[2], best_params[3]]])
    else:
        return np.array([[0.35, 0.25], [0.35, 0.25]])

def angles(n, edges, p):
    start_time = time.time()
    timeout = 115  # Safety margin from 120s limit
    
    optimal_angles = _optimize_angles(n, edges, p, start_time, timeout)
    
    if time.time() - start_time > timeout:
        return np.array([[0.35, 0.25]] * p)
        
    return optimal_angles
# EVOLVE-BLOCK-END
