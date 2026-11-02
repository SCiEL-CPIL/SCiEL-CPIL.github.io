"""Rewritten program: model-and-optimize for Erdos minimum overlap."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START

def _all_overlaps(h):
    """Compute all circular overlaps sum_i h[i]*(1-h[i+k]) for k=0..N-1."""
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, -k)
        overlaps[k] = np.sum(h * (1.0 - shifted))
    return overlaps

def _overlap_max(h, k_max=None):
    """Compute max over shifts k of sum_i h[i]*(1-h[i+k]), circular."""
    if k_max is None:
        k_max = N
    overlaps = _all_overlaps(h)[:k_max]
    return np.max(overlaps)

def _soft_max_overlaps(h, beta=50.0):
    """Smooth surrogate for max using log-sum-exp."""
    overlaps = _all_overlaps(h)
    # Use log-sum-exp for numerical stability
    m = np.max(overlaps)
    if m < -100:
        return m
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def _soft_max_grad(h, beta=50.0):
    """Gradient of soft-max overlap with respect to h."""
    overlaps = _all_overlaps(h)
    m = np.max(overlaps)
    if m < -100:
        return np.zeros_like(h), m
    exp_vals = np.exp(beta * (overlaps - m))
    weights = exp_vals / np.sum(exp_vals)
    
    # d(overlap_k)/dh[i] = (1-h[i+k]) - h[i] * h[i+k-1] for the k-th overlap
    # Actually: overlap_k = sum_i h[i] * (1 - h[(i+k) % N])
    # d(overlap_k)/dh[j] = (1 - h[(j+k) % N]) - h[(j-k) % N]
    
    grad = np.zeros_like(h)
    for k in range(N):
        if weights[k] < 1e-15:
            continue
        dk_dh = (1.0 - np.roll(h, -k)) - np.roll(h, k)
        grad += weights[k] * dk_dh
    return grad, m + np.log(np.sum(exp_vals)) / beta

def params_to_h(params, num_harmonics):
    """Convert harmonic parameters to step heights in [0,1] with sum N/2."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    for m in range(1, num_harmonics + 1):
        a = params[2*(m-1)]
        b = params[2*(m-1)+1]
        h += a * np.sin(2*np.pi*m*i/N) + b * np.cos(2*np.pi*m*i/N)
    
    # Project to [0,1] with sum constraint using iterative water-filling
    h = np.clip(h, 0.0, 1.0)
    for _ in range(20):
        s = np.sum(h)
        if abs(s - N/2.0) < 1e-12:
            break
        # Scale and re-clip
        h = h * (N/2.0) / s
        h = np.clip(h, 0.0, 1.0)
    
    # Final correction to ensure exact sum
    diff = N/2.0 - np.sum(h)
    if abs(diff) > 1e-15:
        # Distribute the difference, being careful not to violate bounds
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        # If still off, adjust remaining elements
        diff2 = N/2.0 - np.sum(h)
        if abs(diff2) > 1e-15:
            idx = np.argmax(np.abs(h - 0.5))
            h[idx] += diff2
            h = np.clip(h, 0.0, 1.0)
    
    return h

def _objective_and_grad(params, num_harmonics, beta=50.0):
    """Objective and gradient for L-BFGS-B."""
    h = params_to_h(params, num_harmonics)
    val = _soft_max_overlaps(h, beta)
    grad_h, _ = _soft_max_grad(h, beta)
    
    # Chain rule: d(params_to_h)/d(params)
    i = np.arange(N)
    grad_params = np.zeros_like(params)
    for m in range(1, num_harmonics + 1):
        a_idx = 2*(m-1)
        b_idx = 2*(m-1)+1
        grad_params[a_idx] = np.sum(grad_h * np.sin(2*np.pi*m*i/N))
        grad_params[b_idx] = np.sum(grad_h * np.cos(2*np.pi*m*i/N))
    
    return val, grad_params

def construct():
    deadline = time.time() + 115.0  # Safety margin
    best = np.full(N, 0.5)
    best_score = _overlap_max(best)
    
    num_harmonics = 5
    n_params = 2 * num_harmonics
    
    # Phase 1: Use L-BFGS-B with smooth surrogate from multiple initializations
    rng = np.random.RandomState(42)
    betas = [10.0, 50.0, 100.0]
    
    for trial in range(8):
        if time.time() > deadline - 30.0:
            break
        
        if trial == 0:
            x0 = np.zeros(n_params)
        else:
            x0 = rng.randn(n_params) * 0.1
        
        for beta in betas:
            if time.time() > deadline - 20.0:
                break
            
            def obj_grad(x):
                val, grad = _objective_and_grad(x, num_harmonics, beta)
                return val, grad
            
            try:
                res = minimize(obj_grad, x0, jac=True, method='L-BFGS-B',
                             options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12})
                if time.time() > deadline - 15.0:
                    break
                
                h_opt = params_to_h(res.x, num_harmonics)
                score = _overlap_max(h_opt)
                if score < best_score - 1e-12:
                    best = h_opt
                    best_score = score
                    x0 = res.x.copy()
            except Exception:
                pass
    
    # Phase 2: Final polish with Nelder-Mead on the nonsmooth objective
    if time.time() < deadline - 10.0:
        # Get current params from best
        # Reconstruct params that give best
        i = np.arange(N)
        # Use least squares to find harmonic coefficients
        A = np.zeros((N, n_params))
        for m in range(1, num_harmonics + 1):
            A[:, 2*(m-1)] = np.sin(2*np.pi*m*i/N)
            A[:, 2*(m-1)+1] = np.cos(2*np.pi*m*i/N)
        
        # Solve for params that best approximate best
        target = best - 0.5
        try:
            params_est, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
        except:
            params_est = np.zeros(n_params)
        
        def objective_nm(x):
            h = params_to_h(x, num_harmonics)
            return _overlap_max(h)
        
        try:
            res = minimize(objective_nm, params_est, method='Nelder-Mead',
                         options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            h_opt = params_to_h(res.x, num_harmonics)
            score = _overlap_max(h_opt)
            if score < best_score - 1e-12:
                best = h_opt
                best_score = score
        except Exception:
            pass
    
    # Ensure sum is exactly N/2
    diff = N/2.0 - np.sum(best)
    if abs(diff) > 1e-15:
        mask = (best > 1e-10) & (best < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            best[mask] += diff / np.sum(mask)
            best = np.clip(best, 0.0, 1.0)
        diff2 = N/2.0 - np.sum(best)
        if abs(diff2) > 1e-15:
            idx = np.argmax(np.abs(best - 0.5))
            best[idx] += diff2
            best = np.clip(best, 0.0, 1.0)
    
    return best

# EVOLVE-BLOCK-END
