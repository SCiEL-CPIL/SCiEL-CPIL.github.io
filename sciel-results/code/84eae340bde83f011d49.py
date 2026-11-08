import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 600
DEADLINE = 130.0  # safety margin before 150s kill

def _autoconv(v):
    """Compute full autoconvolution of v using FFT for speed."""
    n = len(v)
    size = 2 * n - 1
    f = np.fft.rfft(v, n=size)
    ac = np.fft.irfft(f * f.conj(), n=size)
    return ac

def _smooth_max(ac, beta):
    """Smooth approximation of max using log-sum-exp."""
    ac_shifted = ac - np.max(ac)
    return np.log(np.sum(np.exp(beta * ac_shifted))) / beta + np.max(ac)

def _objective_smooth(v, beta):
    """Normalized smooth max of autoconvolution."""
    ac = _autoconv(v)
    smax = _smooth_max(ac, beta)
    total = np.sum(v)
    if total < 1e-12:
        return 1e10
    return smax / (total * total)

def _objective_exact(v):
    """Exact normalized max of autoconvolution."""
    ac = _autoconv(v)
    max_ac = np.max(np.abs(ac))
    total = np.sum(v)
    if total < 1e-12:
        return 1e10
    return max_ac / (total * total)

def construct():
    """
    Returns step heights minimizing the normalized maximum of autoconvolution.
    
    Uses smooth max approximation with gradient-based optimization,
    starting from diverse initial conditions, within a time budget.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_v = np.ones(N) / N
    best_score = _objective_exact(best_v)
    
    # Try multiple initial conditions and betas
    initial_guesses = []
    
    # 1. Uniform
    initial_guesses.append(np.ones(N) / N)
    
    # 2. Triangular (bell-shaped)
    x = np.linspace(-1, 1, N)
    tri = np.maximum(0, 1 - np.abs(x))
    tri = tri / np.sum(tri)
    initial_guesses.append(tri)
    
    # 3. Two-peak
    peak1 = np.zeros(N)
    peak2 = np.zeros(N)
    peak1[:N//4] = 1.0
    peak2[3*N//4:] = 1.0
    two_peak = (peak1 + peak2) / np.sum(peak1 + peak2)
    initial_guesses.append(two_peak)
    
    # 4. Random smooth
    rng = np.random.default_rng(42)
    rand_smooth = np.cumsum(rng.normal(0, 1, N))
    rand_smooth = np.maximum(0, rand_smooth - rand_smooth.min())
    rand_smooth = rand_smooth / np.sum(rand_smooth)
    initial_guesses.append(rand_smooth)
    
    # 5. Cosine window
    cos_w = 0.5 * (1 + np.cos(2 * np.pi * np.arange(N) / N))
    cos_w = np.maximum(0, cos_w)
    cos_w = cos_w / np.sum(cos_w)
    initial_guesses.append(cos_w)
    
    betas = [50, 100, 200]
    
    for v0 in initial_guesses:
        if time.time() > deadline - 10:
            break
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
            
            try:
                def obj(x):
                    return _objective_smooth(x, beta)
                
                def grad(x):
                    # Numerical gradient
                    eps = 1e-7
                    g = np.zeros_like(x)
                    for i in range(N):
                        xp = x.copy()
                        xm = x.copy()
                        xp[i] += eps
                        xm[i] -= eps
                        g[i] = (obj(xp) - obj(xm)) / (2 * eps)
                    return g
                
                result = minimize(
                    obj, v0, method='L-BFGS-B',
                    jac=grad,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < best_score:
                    v_opt = np.maximum(0, result.x)
                    v_opt = v_opt / np.sum(v_opt)
                    score = _objective_exact(v_opt)
                    if score < best_score:
                        best_score = score
                        best_v = v_opt.copy()
                        
            except Exception:
                continue
    
    # Final local refinement with exact objective using small perturbations
    if time.time() < deadline - 5:
        # Simple coordinate descent on the exact objective
        v_cur = best_v.copy()
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(0, N, 3):  # stride to save time
                if time.time() > deadline - 5:
                    break
                for delta in [-0.01, 0.01]:
                    v_try = v_cur.copy()
                    v_try[i] = max(0, v_cur[i] + delta)
                    if np.sum(v_try) < 1e-12:
                        continue
                    v_try = v_try / np.sum(v_try)
                    s = _objective_exact(v_try)
                    if s < best_score:
                        best_score = s
                        v_cur = v_try
                        improved = True
                        break
            if not improved:
                break
        best_v = v_cur
    
    # Ensure nonnegative and normalized
    best_v = np.maximum(0, best_v)
    total = np.sum(best_v)
    if total > 0:
        best_v = best_v / total
    
    return list(best_v)
