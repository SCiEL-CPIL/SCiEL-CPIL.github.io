import numpy as np
from scipy.optimize import minimize
import time

N = 50
W = 1.0 / 25.0  # width of each interval

def _autoconv_max(h):
    """Compute the supremum norm of the autoconvolution of piecewise-constant h."""
    n = len(h)
    # For piecewise-constant functions on uniform grid, the autoconvolution
    # is piecewise-linear. The maximum occurs at grid points or midpoints.
    # We compute (h*h)(x) at all relevant points.
    # (h*h)(x) = sum_i sum_j h[i]*h[j] * overlap_length(x, i, j)
    # where overlap_length depends on the geometry.
    
    # Simpler approach: use the fact that for discrete approximation,
    # the autoconvolution at grid point k is approximately:
    # (h*h)[k] = sum_i h[i] * h[k-i] * W
    # But we need the exact continuous version.
    
    # The exact autoconvolution for piecewise-constant functions:
    # h(t) = h[i] for t in [i*W, (i+1)*W)
    # (h*h)(x) = integral of h(t)h(x-t) dt
    # This is a piecewise-linear function with knots at multiples of W.
    
    # For efficiency, compute at all grid points and midpoints
    # The maximum of a piecewise-linear function occurs at a knot.
    
    # Grid points: x = k*W for k = -n+1, ..., n-1
    # Actually, the support of h*h is [-n*W, n*W]
    # Knots are at x = m*W for integer m
    
    # Let's compute (h*h)(x) at x = m*W for m = -(n-1), ..., n-1
    # and also at midpoints x = (m+0.5)*W
    
    # For x = m*W:
    # (h*h)(m*W) = sum_i h[i] * h[m-i] * W
    # (this is the discrete convolution times W)
    
    # But we need to be more careful. Let's use the exact formula.
    
    # Actually, for the purpose of optimization, let's use a numerical approach
    # that's fast enough.
    
    # Compute autoconvolution via FFT for speed, then find max
    # Pad to avoid circular convolution
    n_pad = 2 * n + 1
    h_pad = np.zeros(n_pad)
    h_pad[:n] = h
    h_conv = np.fft.ifft(np.fft.fft(h_pad) * np.fft.fft(h_pad))
    h_conv = np.real(h_conv)
    # The autoconvolution at grid point k is h_conv[k] * W
    # But we need the supremum, which for piecewise-linear is at knots or midpoints
    
    # For piecewise-constant functions, (h*h)(x) is piecewise-linear with slope changes
    # at x = m*W. The maximum occurs at a knot x = m*W.
    
    # So max(h*h) = max_k h_conv[k] * W
    return np.max(h_conv) * W

def _ratio(h):
    """Compute the ratio ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    l2_sq = np.sum(h**2) * W
    l1 = np.sum(h) * W
    autoconv_max = _autoconv_max(h)
    if l1 * autoconv_max < 1e-30:
        return 0.0
    return l2_sq / (l1 * autoconv_max)

def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Try a few initializations
    initializations = [
        np.ones(N),
        np.array([1.0 if i % 2 == 0 else 0.1 for i in range(N)]),
        np.array([1.0 if i < N//2 else 0.1 for i in range(N)]),
        np.random.RandomState(42).rand(N) * 2,
    ]
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        h0 = np.maximum(h0, 0.0)  # ensure nonnegative
        
        def neg_ratio(x):
            x = np.maximum(x, 0.0)
            r = _ratio(x)
            return -r if r > 0 else 0.0
        
        try:
            res = minimize(neg_ratio, h0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            h_opt = np.maximum(res.x, 0.0)
            score = _ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 5.0:
        try:
            def neg_ratio_precise(x):
                x = np.maximum(x, 0.0)
                r = _ratio(x)
                return -r if r > 0 else 0.0
            
            res = minimize(neg_ratio_precise, best_h, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            h_opt = np.maximum(res.x, 0.0)
            score = _ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return list(best_h)
