import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _objective(h):
    """Compute the value we want to minimize."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def _objective_and_grad(h):
    """Objective and gradient for the max correlation term."""
    h = np.clip(h, 0.0, 1.0)
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode="full")
    max_idx = np.argmax(corr)
    value = corr[max_idx] / N * 2.0
    
    grad = np.zeros(N)
    shift = max_idx - (N - 1)
    
    for j in range(N):
        g = 1.0
        if 0 <= j + shift < N:
            g -= h[j + shift]
        if 0 <= j - shift < N:
            g -= h[j - shift]
        grad[j] = g
    
    grad = grad / N * 2.0
    return value, grad

def _project(h):
    """Project to feasible set: h in [0,1], sum(h) = N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(50):
        current_sum = np.sum(h)
        diff = target - current_sum
        if abs(diff) < 1e-14:
            break
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
        else:
            break
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Multiple random restarts with L-BFGS-B
    n_restarts = 40
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            h0 = np.full(N, 0.5)
        elif r < 5:
            # Structured initializations
            if r == 1:
                h0 = np.tile([0.0, 1.0], N // 2)
            elif r == 2:
                h0 = np.tile([0.0, 0.0, 1.0, 1.0], N // 4)
            elif r == 3:
                h0 = np.sin(np.linspace(0, 4*np.pi, N)) * 0.5 + 0.5
            else:
                h0 = np.cos(np.linspace(0, 6*np.pi, N)) * 0.5 + 0.5
            h0 = _project(h0)
        else:
            h0 = rng.random(N)
            h0 = _project(h0)
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x)[0],
                h0,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            h_opt = _project(res.x)
            val = _objective(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Fine-tune from best solution
    if time.time() < deadline - 10.0:
        try:
            for _ in range(10):
                if time.time() > deadline:
                    break
                res = minimize(
                    lambda x: _objective_and_grad(x)[0],
                    best_h,
                    jac=lambda x: _objective_and_grad(x)[1],
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-11}
                )
                h_opt = _project(res.x)
                val = _objective(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Local perturbation search
    if time.time() < deadline - 5.0:
        for _ in range(200):
            if time.time() > deadline:
                break
            h_pert = best_h + rng.normal(0, 0.01, N)
            h_pert = _project(h_pert)
            val = _objective(h_pert)
            if val < best_val:
                best_val = val
                best_h = h_pert.copy()
                # Refine
                try:
                    res = minimize(
                        lambda x: _objective_and_grad(x)[0],
                        best_h,
                        jac=lambda x: _objective_and_grad(x)[1],
                        method='L-BFGS-B',
                        bounds=[(0.0, 1.0)] * N,
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    h_opt = _project(res.x)
                    val = _objective(h_opt)
                    if val < best_val:
                        best_val = val
                        best_h = h_opt.copy()
                except Exception:
                    pass
    
    best_h = _project(best_h)
    return best_h
