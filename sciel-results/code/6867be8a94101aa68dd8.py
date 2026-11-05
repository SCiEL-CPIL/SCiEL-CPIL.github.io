import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400
_n = np.arange(N)
_DEADLINE = 140.0  # safety margin under 150s limit

def _value(h):
    """Compute the objective value (lower is better)."""
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _make_h_cheb(coeffs, K):
    """
    Construct h as linear combination of Chebyshev polynomials.
    h[n] = sum_{j=0}^{K-1} coeffs[j] * T_j(x_n)
    where x_n = linspace(-1, 1, N)
    """
    x = np.linspace(-1, 1, N)
    # Precompute Chebyshev basis
    basis = np.zeros((K, N))
    basis[0] = 1.0
    if K > 1:
        basis[1] = x
    for j in range(2, K):
        basis[j] = 2.0 * x * basis[j-1] - basis[j-2]
    h = basis.T @ coeffs
    return h

def _obj_cheb(coeffs, K):
    """Objective function for Chebyshev coefficient optimization."""
    h = _make_h_cheb(coeffs, K)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12
    c = np.convolve(h, h)
    max_c = np.max(c)
    val = 2.0 * N * max_c / (s * s)
    return abs(val)

def _make_h_fourier(coeffs, M):
    """
    Band-limited real signal with unit sum:
    h[n] = (1 + sum_{j>=1} a_j cos(2 pi j n/N) + b_j sin(2 pi j n/N)) / N
    """
    h = np.full(N, 1.0 / N)
    for j in range(1, M + 1):
        a = coeffs[2 * (j - 1)]
        b = coeffs[2 * (j - 1) + 1]
        h += (a * np.cos(2 * np.pi * j * _n / N) + b * np.sin(2 * np.pi * j * _n / N)) / N
    return h

def _obj_fourier(coeffs, M):
    h = _make_h_fourier(coeffs, M)
    s = h.sum()
    if abs(s) < 1e-12:
        return 1e12
    c = np.convolve(h, h)
    return abs(2.0 * N * c.max() / (s * s))

def _subgradient_optimize(h_init, max_iter=5000, eta=0.05, mu=0.9):
    """
    Subgradient optimization with momentum for refining a candidate.
    Minimizes max(conv(h,h)) subject to sum(h)=1 (via normalization).
    """
    h = h_init.copy().astype(float)
    s = np.sum(h)
    if abs(s) < 1e-12:
        h += np.random.normal(0, 1e-6, N)
        s = np.sum(h)
    h = h / s
    
    v = np.zeros_like(h)
    best_local_val = np.inf
    best_local_h = h.copy()
    t0 = time.time()
    
    for it in range(max_iter):
        if time.time() - t0 > 10:  # Limit each subgradient run to 10s
            break
            
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        k_max = np.argmax(conv)
        
        local_val = 2.0 * N * max_conv
        if local_val < best_local_val:
            best_local_val = local_val
            best_local_h = h.copy()
            
        # Compute subgradient of max_conv wrt h
        # d c_k / d h_m = 2 * h_{k-m} if 0 <= k-m < N else 0
        idx = k_max - np.arange(N)
        g = np.zeros(N)
        valid = (idx >= 0) & (idx < N)
        g[valid] = 2.0 * h[idx[valid]]
        
        v = mu * v - eta * g
        h = h + v
        
        # Normalize to sum=1
        s = np.sum(h)
        if abs(s) < 1e-12:
            h += np.random.normal(0, 1e-6, N)
            s = np.sum(h)
        h = h / s
        
        # Adaptive step size: reduce if value increases
        if it % 50 == 0:
            cur_val = 2.0 * N * np.max(np.convolve(h, h))
            if cur_val > best_local_val * 1.5:
                eta *= 0.5
            elif cur_val < best_local_val * 0.95:
                eta = min(eta * 1.1, 0.2)
                
    return best_local_h

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    
    best_h = np.ones(N).astype(float)
    best_val = 2.0
    
    def _update(h):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 1: Global search using Chebyshev basis (low-dimensional, smooth)
    # Try different numbers of basis functions
    for K in [3, 5, 7, 10]:
        if time.time() - start_time > _DEADLINE - 30:
            break
        
        dim = K
        # Bounds: T_0 coefficient should be dominant to keep sum(h) nonzero
        bounds = [(0.5, 5.0)] + [(-3.0, 3.0)] * (K - 1)
        
        # Estimate max iterations based on remaining time
        remaining = _DEADLINE - 30 - (time.time() - start_time)
        if remaining < 5:
            continue
        maxiter = max(20, int(remaining * 1.5))
        popsize = max(10, min(40, 35 - K))
        
        try:
            res = differential_evolution(
                _obj_cheb, bounds, args=(K,),
                maxiter=maxiter, popsize=popsize,
                tol=0.01, mutation=(0.5, 1.0), recombination=0.7,
                seed=42, polish=False,
                init='sobol' if dim <= 20 else 'latinhypercube'
            )
            if res.x is not None:
                h = _make_h_cheb(res.x, K)
                _update(h)
                
                # Local polish with Nelder-Mead
                if time.time() - start_time < _DEADLINE - 15:
                    try:
                        r2 = minimize(_obj_cheb, res.x, args=(K,), method='Nelder-Mead',
                                      options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                        if r2.x is not None:
                            h2 = _make_h_cheb(r2.x, K)
                            _update(h2)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Phase 2: Global search using Fourier basis (alternative smooth family)
    for M in [2, 4, 6, 8]:
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        dim = 2 * M
        bounds = [(-2.0, 2.0)] * dim
        
        remaining = _DEADLINE - 20 - (time.time() - start_time)
        if remaining < 5:
            continue
        maxiter = max(20, int(remaining * 1.0))
        popsize = max(10, min(30, 25 - M))
        
        try:
            res_f = differential_evolution(
                _obj_fourier, bounds, args=(M,),
                maxiter=maxiter, popsize=popsize,
                tol=0.02, mutation=(0.4, 1.0), recombination=0.8,
                seed=123, polish=False,
                init='sobol' if dim <= 20 else 'latinhypercube'
            )
            if res_f.x is not None:
                h_f = _make_h_fourier(res_f.x, M)
                _update(h_f)
        except Exception:
            pass
    
    # Phase 3: Refine the best candidate using subgradient descent
    # This allows exploring the full 400-dimensional space locally
    if time.time() - start_time < _DEADLINE - 10:
        # Try multiple refinements from the best solution found
        for attempt in range(3):
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            # Add small noise to escape local minima
            noise_scale = 0.01 * (attempt + 1)
            h_perturbed = best_h + np.random.normal(0, noise_scale, N)
            
            h_refined = _subgradient_optimize(h_perturbed, max_iter=3000, eta=0.02, mu=0.85)
            _update(h_refined)
            
            # Also try refining without noise
            if attempt == 0:
                h_refined2 = _subgradient_optimize(best_h, max_iter=3000, eta=0.01, mu=0.9)
                _update(h_refined2)
    
    # Final validation
    final_val = _value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N).astype(float)
    
    return best_h
