"""Improved program: global search via Differential Evolution + log-space L-BFGS-B refinement."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 50

def _autoconv_max(h):
    """Compute the maximum of the (discrete) autoconvolution of h.
    For a sequence h[0..N-1], the autoconvolution c[k] = sum_i h[i]*h[i+k]
    but for the step function interpretation, we use the linear convolution
    and take the max of the symmetric part.
    
    Actually, for a step function on equal intervals, the continuous autoconvolution
    at point x is integral f(t)f(x-t) dt. For step functions with height h_i on
    interval [i*dx, (i+1)*dx], this is a piecewise linear function.
    
    A common discrete approximation: c[k] = sum_i h[i] * h[i+k] for lag k,
    but the full convolution (not just correlation) gives:
    conv[k] = sum_i h[i] * h[k-i] (with appropriate bounds)
    
    The supremum of the continuous autoconvolution of a nonnegative step function
    is bounded above by max_i h[i]^2 * dx + ... but let's use the standard
    discrete convolution approach.
    
    For nonnegative h, the autoconvolution (correlation) c[k] = sum_i h[i]*h[i+k]
    is maximized at k=0, giving sum h[i]^2. But the task says "autoconvolution"
    which is (f*f)(x) = integral f(t)f(x-t)dt.
    
    For step functions, (f*f) is a piecewise linear function. Its maximum occurs
    at some point. A discrete approximation using convolution:
    
    Let's use: the continuous autoconvolution of step function with heights h_i
    on intervals of width dx=1/50 in [-1/4,1/4].
    
    The integral (f*f)(x) = sum_i sum_j h_i * h_j * length of intersection
    of [i*dx,(i+1)*dx] and [(x-j*dx),(x-(j+1)*dx)].
    
    This is complex. A simpler valid lower bound / proxy: use the discrete
    convolution max. Many formulations of this problem use:
    
    score = (sum h_i^2) / (sum h_i * max_k (sum_i h_i * h_{i+k}))
    
    where max_k is over all lags k. But for nonnegative h, the autoconvolution
    sup is at least max_i h_i^2 * dx, and the discrete max of correlation is
    sum h_i^2 (at lag 0).
    
    Let me reconsider: the task says "squared L2 norm to L1 norm times supremum norm of autoconvolution".
    
    So: score = ||f||_2^2 / (||f||_1 * ||f*f||_inf)
    
    For step functions:
    ||f||_2^2 = sum h_i^2 * dx
    ||f||_1 = sum h_i * dx
    ||f*f||_inf = sup_x (f*f)(x)
    
    So score = (sum h_i^2 * dx) / (sum h_i * dx * sup(f*f))
             = (sum h_i^2) / (sum h_i * sup(f*f)/dx)
    
    This is scale-invariant in a specific way. Let me just compute the continuous
    autoconvolution sup numerically for the step function.
    
    Actually, for efficiency, let's use the fact that for a step function on [a,b]
    with N equal subintervals, the autoconvolution (f*f)(x) can be computed.
    
    Let dx = 1/50 (since interval is [-1/4,1/4], length = 1/2, so dx = 1/100? 
    Wait: [-1/4, 1/4] has length 1/2. 50 intervals, so dx = 1/100.
    
    Let me use a direct numerical approach: evaluate the autoconvolution on a
    fine grid and take the max.
    """
    dx = 0.5 / N  # interval [-1/4, 1/4] has length 0.5, 50 steps
    # Step function: f(x) = h[i] for x in [x_i, x_i + dx) where x_i = -0.25 + i*dx
    # (f*f)(x) = integral f(t) f(x-t) dt
    # This is the convolution of f with itself.
    
    # For efficiency, use the discrete convolution approximation:
    # The continuous convolution of two step functions is a piecewise linear function.
    # Its values at the grid points and midpoints can be computed.
    
    # Simpler: use numpy convolve on the heights with appropriate scaling
    # The continuous autoconvolution sup is approximately:
    # max over k of (sum_i h[i]*h[i+k]) * dx  for the correlation, but
    # autoconvolution is different from correlation.
    
    # Let me use: conv = np.convolve(h, h, mode='full') * dx
    # This gives the values of the continuous convolution at the grid points
    # if we interpret it correctly. Actually, np.convolve(h,h) gives
    # sum_i h[i]*h[k-i] which is the cross-correlation reversed.
    
    # For autoconvolution (f*f)(x) = int f(t)f(x-t)dt, with f being step:
    # If h has N elements, the convolution has 2N-1 elements.
    # (f*f) is a continuous piecewise linear function. Its max is at one of
    # the "knots" which are at x = (i - j)*dx for i,j in range.
    
    # A practical approach: compute the discrete convolution and use the max.
    # For nonnegative h, the max of the continuous autoconvolution is achieved
    # at x = k*dx for some integer k, and equals:
    # sum_{i} h[i] * h[k-i] * dx  (with h out of range = 0)
    
    # So: c = np.convolve(h, h) * dx, and sup = max(c)
    c = np.convolve(h, h, mode='full') * dx
    return np.max(c)

def _score(h):
    """Compute the ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf)"""
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0)  # ensure nonnegative
    dx = 0.5 / N
    l2_sq = np.sum(h**2) * dx
    l1 = np.sum(h) * dx
    ac_max = _autoconv_max(h)
    if l1 * ac_max < 1e-30:
        return 0.0
    return l2_sq / (l1 * ac_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s
    
    best_h = None
    best_score = 0.0
    
    # Strategy: Differential Evolution for global search, then L-BFGS-B refinement
    # Work in log-space for L-BFGS-B to maintain nonnegativity
    
    bounds = [(1e-4, 10.0)] * N
    
    # First run: DE with moderate population
    try:
        result = differential_evolution(
            lambda x: -_score(x),
            bounds=bounds,
            maxiter=30,
            popsize=10,
            seed=42,
            tol=1e-4,
            polish=False,
            callback=lambda xk, fk: (_ for _ in ()).throw(TimeoutError) if time.time() > deadline - 30 else None
        )
        if time.time() > deadline - 20:
            h = result.x
            s = _score(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
            return [max(0, v) for v in best_h] if best_h is not None else [1.0] * N
        
        h = result.x
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        
        # L-BFGS-B refinement in log-space
        if time.time() < deadline - 15:
            def neg_score_log(z):
                h = np.exp(z)
                return -_score(h)
            
            z0 = np.log(np.maximum(best_h, 1e-6))
            res = minimize(neg_score_log, z0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < neg_score_log(z0):
                h_refined = np.exp(res.x)
                s_refined = _score(h_refined)
                if s_refined > best_score:
                    best_score = s_refined
                    best_h = h_refined.copy()
    except Exception:
        pass
    
    # If time remains, do a second DE run from a different starting point
    if time.time() < deadline - 40 and best_h is not None:
        try:
            # Perturb the best solution
            rng = np.random.RandomState(123)
            h_perturb = best_h * (1 + 0.3 * rng.randn(N))
            h_perturb = np.maximum(h_perturb, 1e-4)
            
            # Local search from perturbed point
            def neg_score_direct(x):
                return -_score(x)
            
            bounds2 = [(1e-4, 10.0)] * N
            res2 = minimize(neg_score_direct, h_perturb, method='L-BFGS-B',
                           bounds=bounds2,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res2.fun < -best_score:
                h2 = np.maximum(res2.x, 0)
                s2 = _score(h2)
                if s2 > best_score:
                    best_score = s2
                    best_h = h2.copy()
        except Exception:
            pass
    
    # Final safety: return best found or default
    if best_h is None:
        return [1.0] * N
    
    h_final = np.maximum(best_h, 0)
    # Ensure the result is a list of floats
    return [float(v) for v in h_final]

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
