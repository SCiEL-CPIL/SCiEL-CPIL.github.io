import numpy as np
import time
from scipy.optimize import minimize

N = 600

def _build_f_from_params(params):
    """
    params: array of shape (3*K,) where K is number of bumps.
    Each bump has (center, width, amplitude).
    We construct f as a sum of triangular bumps, normalized to sum to 1.
    """
    K = len(params) // 3
    f = np.zeros(N)
    x = np.linspace(-0.25, 0.25, N)
    
    for k in range(K):
        center = params[3*k]
        width = params[3*k + 1]
        amp = params[3*k + 2]
        
        # Triangular bump: max(0, 1 - |x-center|/width) * amp
        tri = np.maximum(0.0, 1.0 - np.abs(x - center) / (width + 1e-10))
        f += amp * tri
    
    # Ensure nonnegativity and normalize
    f = np.maximum(f, 0.0)
    total = np.sum(f)
    if total > 1e-15:
        f /= total
    else:
        f = np.ones(N) / N
    
    return f

def _objective(params):
    """
    Compute max of autoconvolution of f built from params.
    """
    f = _build_f_from_params(params)
    c = np.convolve(f, f)
    return np.max(c)

def _logsumexp_max(c, beta=100.0):
    """
    Smooth approximation to max(c) using log-sum-exp.
    """
    c_max = np.max(c)
    if c_max <= 0:
        return 0.0
    return c_max + np.log(np.sum(np.exp(beta * (c - c_max)))) / beta

def _objective_smooth(params, beta=100.0):
    """
    Smooth objective using log-sum-exp approximation of max.
    """
    f = _build_f_from_params(params)
    c = np.convolve(f, f)
    return _logsumexp_max(c, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety
    
    # Try different numbers of bumps
    best_f = np.ones(N) / N
    best_score = np.max(np.convolve(best_f, best_f))
    
    # Initial parameters: start with symmetric bumps
    # For K bumps, place them symmetrically
    K_values = [2, 3, 4, 5]
    
    for K in K_values:
        if time.time() > deadline - 5:
            break
        
        # Initial guess: K bumps spread across [-0.2, 0.2]
        centers = np.linspace(-0.15, 0.15, K)
        widths = np.full(K, 0.05)
        amps = np.full(K, 1.0)
        
        params0 = np.concatenate([centers, widths, amps])
        
        # Bounds: centers in [-0.24, 0.24], widths in [0.01, 0.25], amps in [0.01, 5]
        bounds = []
        for k in range(K):
            bounds.append((-0.24, 0.24))  # center
            bounds.append((0.01, 0.25))    # width
            bounds.append((0.01, 5.0))     # amp
        
        # Use Nelder-Mead on the smooth objective
        try:
            result = minimize(
                _objective_smooth,
                params0,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            f_candidate = _build_f_from_params(result.x)
            score = np.max(np.convolve(f_candidate, f_candidate))
            
            if score < best_score:
                best_score = score
                best_f = f_candidate.copy()
        except Exception:
            pass
        
        # Also try Nelder-Mead on the raw max objective (non-smooth)
        if time.time() < deadline - 10:
            try:
                result2 = minimize(
                    _objective,
                    params0,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                
                f_candidate2 = _build_f_from_params(result2.x)
                score2 = np.max(np.convolve(f_candidate2, f_candidate2))
                
                if score2 < best_score:
                    best_score = score2
                    best_f = f_candidate2.copy()
            except Exception:
                pass
    
    # Fine-tune with direct optimization on the final f using a few gradient steps
    if time.time() < deadline - 5:
        f = best_f.copy()
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            
            c = np.convolve(f, f)
            max_val = np.max(c)
            max_idx = np.argmax(c)
            
            contrib = np.zeros(N)
            for i in range(N):
                j = max_idx - i
                if 0 <= j < N:
                    contrib[i] = f[j]
            
            contrib_sum = np.sum(contrib)
            if contrib_sum > 0:
                contrib_norm = contrib / contrib_sum
            else:
                contrib_norm = np.ones(N) / N
            
            alpha = 0.05
            f = f * (1 - alpha * contrib_norm)
            f = np.clip(f, 0, None)
            total = np.sum(f)
            if total > 1e-15:
                f = f / total
            
            current_score = np.max(np.convolve(f, f))
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
    
    return list(best_f)
