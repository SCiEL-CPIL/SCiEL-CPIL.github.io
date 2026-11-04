import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
W = 1.0 / 25.0  # width of each interval

def _autoconv_max(h):
    """Compute the supremum norm of the autoconvolution of piecewise-constant h."""
    n = len(h)
    # For piecewise-constant functions on uniform grid, the autoconvolution
    # is piecewise-linear. The maximum occurs at grid points.
    # (h*h)(x) = sum_i sum_j h[i]*h[j] * overlap_length(x, i, j)
    
    # Using FFT for efficiency
    n_pad = 2 * n + 1
    h_pad = np.zeros(n_pad)
    h_pad[:n] = h
    h_conv = np.fft.ifft(np.fft.fft(h_pad) * np.fft.fft(h_pad))
    h_conv = np.real(h_conv)
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
    deadline = time.time() + 145.0  # safety margin before 150s limit
    
    # Start with a good initial guess
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Try various structural initializations
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Sparse: single peak
    for peak_pos in [0, N//4, N//2, 3*N//4, N-1]:
        h = np.zeros(N)
        h[peak_pos] = 1.0
        initializations.append(h)
    
    # Sparse: two peaks
    for p1 in range(0, N, N//10):
        for p2 in range(p1+5, N, N//10):
            h = np.zeros(N)
            h[p1] = 1.0
            h[p2] = 1.0
            initializations.append(h)
    
    # Sparse: random positions
    rng = np.random.RandomState(42)
    for _ in range(20):
        k = rng.randint(1, 10)  # 1 to 9 non-zero entries
        positions = rng.choice(N, k, replace=False)
        h = np.zeros(N)
        h[positions] = rng.uniform(0.5, 1.5, k)
        initializations.append(h)
    
    # Multi-scale: alternating patterns
    for period in [2, 3, 4, 5]:
        h = np.array([1.0 if i % period == 0 else 0.1 for i in range(N)])
        initializations.append(h)
    
    # Gaussian-like
    x = np.linspace(0, 1, N)
    for center in [0.2, 0.5, 0.8]:
        for width in [0.05, 0.1, 0.2]:
            h = np.exp(-0.5 * ((x - center) / width)**2)
            initializations.append(h)
    
    # Evaluate all initializations
    for h0 in initializations:
        if time.time() > deadline - 10.0:
            break
        h0 = np.maximum(h0, 0.0)
        score = _ratio(h0)
        if score > best_score:
            best_score = score
            best_h = h0.copy()
    
    # Local optimization from best starting point
    if time.time() < deadline - 5.0:
        def neg_ratio(x):
            x = np.maximum(x, 0.0)
            r = _ratio(x)
            return -r if r > 0 else 1e10
        
        # Try multiple local optimizations from different starting points
        local_starts = [best_h]
        # Add a few more diverse starts
        rng = np.random.RandomState(123)
        for _ in range(5):
            k = rng.randint(1, 8)
            positions = rng.choice(N, k, replace=False)
            h = np.zeros(N)
            h[positions] = rng.uniform(0.5, 1.5, k)
            local_starts.append(h)
        
        for h_start in local_starts:
            if time.time() > deadline - 5.0:
                break
            h_start = np.maximum(h_start, 0.0)
            try:
                res = minimize(neg_ratio, h_start, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                h_opt = np.maximum(res.x, 0.0)
                score = _ratio(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
            except Exception:
                pass
        
        # Final polish
        if time.time() < deadline - 2.0:
            try:
                res = minimize(neg_ratio, best_h, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
                h_opt = np.maximum(res.x, 0.0)
                score = _ratio(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    return list(best_h)
