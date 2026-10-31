import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * sup|h*h|).
    h is a nonnegative array of length N.
    Returns the score (higher is better).
    """
    h = np.asarray(h, dtype=float)
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    # Autoconvolution: h * h (discrete convolution)
    # For step functions on equal intervals, the autoconvolution at lag k
    # is sum_i h[i] * h[i+k] (for full convolution, but we need the sup norm)
    # Actually, the autoconvolution of a step function f(x) = h[i] on interval i
    # is (f*f)(x) = integral f(y) f(x-y) dy
    # For discrete steps on equal intervals of width w=1/50 over [-1/4, 1/4]:
    # The convolution at a point is a piecewise linear function.
    # The sup norm of the autoconvolution can be computed from the discrete
    # autocorrelation values.
    
    # The autoconvolution of the step function:
    # f(x) = h[i] for x in [x_i, x_{i+1}) where x_i = -1/4 + i/N
    # (f*f)(x) = integral f(y)f(x-y) dy
    # 
    # For a step function, the convolution is a piecewise quadratic (or linear)
    # function. The maximum occurs either at the centers of the intervals or
    # at the boundaries.
    #
    # A simpler approach: the autoconvolution evaluated at the grid points
    # (centers of intervals) gives the discrete autocorrelation.
    # The sup norm is at least the max of the discrete autocorrelation.
    # For step functions, the sup of the continuous autoconvolution equals
    # the max over all x of the convolution.
    
    # Let's compute the full discrete autocorrelation:
    # r[k] = sum_i h[i] * h[i+k] for k = -(N-1), ..., N-1
    # But for the sup norm of the continuous autoconvolution, we need to be more careful.
    
    # Actually, for a step function with N steps, the autoconvolution is a
    # piecewise quadratic function. The maximum can be found by evaluating
    # at critical points.
    
    # A practical approach: the sup norm of the autoconvolution of a step
    # function is well-approximated by the maximum of the discrete
    # autocorrelation values, especially when N is large.
    
    # Let's compute the full linear autocorrelation:
    r = np.correlate(h, h, mode='full')
    # r has length 2N-1, with r[N-1] being the zero-lag correlation
    sup_auto = np.max(r)
    
    if sup_auto <= 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_auto)


def _neg_score(h):
    return -_score(h)


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s limit
    
    # Start with a Gaussian-like profile
    x = np.linspace(-0.5, 0.5, N)
    sigma = 0.2
    h0 = np.exp(-0.5 * (x / sigma) ** 2)
    h0 = np.maximum(h0, 0.0)
    
    # Normalize
    h0 = h0 / np.sum(h0) * N  # Scale so sum is approximately N
    
    best_h = h0.copy()
    best_score = _score(h0)
    
    # Local refinement using L-BFGS-B
    constraints = []
    bounds = [(0.0, None)] * N
    
    def objective(h):
        return -_score(h)
    
    # Multiple random restarts with different initial shapes
    rng = np.random.default_rng(42)
    
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            h_start = h0.copy()
        elif restart == 1:
            # Triangular
            h_start = np.maximum(1 - np.abs(x) * 2, 0)
            h_start = np.maximum(h_start, 0.01)
        elif restart == 2:
            # Uniform
            h_start = np.ones(N)
        elif restart == 3:
            # Cosine bell
            h_start = np.maximum(np.cos(np.pi * x), 0)
            h_start = np.maximum(h_start, 0.01)
        elif restart == 4:
            # Two peaks
            h_start = np.exp(-0.5 * ((x - 0.2) / 0.1) ** 2) + np.exp(-0.5 * ((x + 0.2) / 0.1) ** 2)
            h_start = np.maximum(h_start, 0.01)
        else:
            # Random smooth start
            h_start = np.ones(N)
            for _ in range(10):
                center = rng.integers(0, N)
                width = rng.uniform(0.1, 0.3)
                amp = rng.uniform(-0.5, 0.5)
                h_start += amp * np.exp(-0.5 * ((np.arange(N) - center) / width) ** 2)
            h_start = np.maximum(h_start, 0.0)
        
        # Scale to have reasonable magnitude
        h_start = h_start / np.sum(h_start) * N
        
        try:
            result = minimize(
                objective,
                h_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < objective(h_start):
                h_opt = np.maximum(result.x, 0.0)
                s = _score(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish: try a very tight optimization from the best point
    if time.time() < deadline - 2.0 and best_score > 0:
        try:
            result = minimize(
                objective,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
            )
            if result.fun < objective(best_h):
                h_opt = np.maximum(result.x, 0.0)
                s = _score(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    return list(best_h)
# EVOLVE-BLOCK-END
