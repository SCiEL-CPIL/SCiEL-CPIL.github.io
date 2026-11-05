"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_value(h):
    """Compute the exact objective value."""
    C = np.convolve(h, h, 'full')
    S = np.sum(h)
    if abs(S) < 1e-12:
        return np.inf
    return abs(2 * N * np.max(C) / (S * S))

def _objective_and_grad(h, tau=1.0):
    """Compute objective and gradient using log-sum-exp surrogate."""
    C = np.convolve(h, h, 'full')  # length 2N-1
    S = np.sum(h)
    
    if abs(S) < 1e-12:
        return 1e10, np.ones(N)
    
    # Log-sum-exp for max(C)
    C_max = np.max(C)
    C_shifted = C - C_max
    exp_C = np.exp(C_shifted / tau)
    sum_exp = np.sum(exp_C)
    LSE = C_max + tau * np.log(sum_exp)
    
    val = 2.0 * N * LSE / (S * S)
    
    # Gradient weights
    w = exp_C / sum_exp  # length 2N-1
    
    # grad[j] = sum_m w[m]*(h[m-j] + h[j-m])
    m = np.arange(2*N-1)
    j = np.arange(N)
    
    mj = m[:, None] - j[None, :]
    jm = j[None, :] - m[:, None]
    
    h_mj = np.where((mj >= 0) & (mj < N), h[mj], 0.0)
    h_jm = np.where((jm >= 0) & (jm < N), h[jm], 0.0)
    
    grad_lse = np.sum(w[:, None] * (h_mj + h_jm), axis=0)
    
    # d/dh [2N*LSE/S^2] = 2N*(grad_lse*S - 2*LSE)/S^3
    grad_obj = 2.0 * N * (grad_lse * S - 2.0 * LSE) / (S**3)
    
    return val, grad_obj


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N).copy()
    best_val = _compute_value(best_h)
    
    np.random.seed(12345)
    
    # Different tau values and initializations
    configs = []
    for tau in [0.3, 0.7, 1.5]:
        for init_type in range(4):
            configs.append((tau, init_type))
    
    for tau, init_type in configs:
        if time.time() > deadline - 25:
            break
            
        if init_type == 0:
            h0 = np.ones(N)
        elif init_type == 1:
            freq = np.random.randint(1, 20)
            h0 = np.cos(2 * np.pi * freq * np.arange(N) / N)
        elif init_type == 2:
            h0 = np.random.randn(N)
        else:
            h0 = np.ones(N) + 0.5 * np.random.randn(N)
        
        if abs(np.sum(h0)) < 1e-6:
            h0 += 1.0
        
        def obj_and_grad(x):
            return _objective_and_grad(x, tau)
        
        try:
            result = minimize(
                obj_and_grad,
                h0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-6, 'maxfun': 500}
            )
            
            h_candidate = result.x
            if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-6:
                exact_val = _compute_value(h_candidate)
                if exact_val < best_val:
                    best_val = exact_val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Post-processing: try to reduce peak by local adjustments
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            # Random perturbation and local optimize
            h_try = best_h + 0.01 * np.random.randn(N)
            if abs(np.sum(h_try)) < 1e-6:
                continue
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, tau=1.0),
                    h_try,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'gtol': 1e-5}
                )
                if np.all(np.isfinite(result.x)) and abs(np.sum(result.x)) > 1e-6:
                    v = _compute_value(result.x)
                    if v < best_val:
                        best_val = v
                        best_h = result.x.copy()
            except:
                pass
    
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-6:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
