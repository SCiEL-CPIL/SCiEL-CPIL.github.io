import numpy as np
import time
import scipy.optimize as opt

N = 600
# Precompute FFT size for linear convolution of length N with itself
# Result length is 2*N - 1. We need an FFT size >= 2*N - 1.
FFT_SIZE = 1 << (2*N - 1).bit_length()

# Precompute a fine grid for accurate evaluation of the peak
# The step function is defined on [-1/4, 1/4] with N intervals.
# Interval width w = 0.5 / N.
# To accurately capture the peak of the autoconvolution, which is a piecewise linear function,
# we evaluate on a finer grid.
# The autoconvolution of two step functions with N steps has at most 2N-1 linear segments.
# The peak will be at one of the vertices or within a segment.
# Using a fine grid of size FINE_N is a robust way to approximate the max.
# Let's use FINE_N = 10 * N to be safe, but keep FFT size manageable.
# Actually, the exact max of the autoconvolution of a step function occurs at the boundaries of the steps.
# So we can just evaluate at the boundaries.
# The boundaries are at -1/4 + i * w for i=0..N.
# The autoconvolution g(t) = integral f(x) f(t-x) dx.
# For step functions, g(t) is piecewise quadratic? No, piecewise linear if we consider the overlap length.
# Wait, f is a step function. f(x) = h_i for x in [x_i, x_{i+1}).
# Then g(t) is the integral of f(x)f(t-x) dx.
# This is the convolution of two step functions.
# The result is a piecewise quadratic function?
# Let's check.
# If f is constant on an interval, the convolution with another step function results in a function that is piecewise quadratic?
# No. The convolution of two indicator functions of intervals is a triangle function (piecewise linear).
# The convolution of two step functions (sums of indicator functions) is a sum of triangle functions.
# So g(t) is piecewise linear.
# The maximum of a piecewise linear function occurs at a vertex.
# The vertices are at the points where the slope changes.
# The slope of g(t) changes when the boundaries of the intervals of f(x) and f(t-x) align.
# This happens when t - x_{i+1} = x_j or t - x_i = x_j, etc.
# Basically, the vertices are at t = x_i + x_j.
# x_i = -1/4 + i * w.
# So vertices are at -1/2 + (i+j)w.
# i, j in 0..N-1.
# So i+j ranges from 0 to 2N-2.
# So there are 2N-1 vertices.
# We can compute the exact value of g(t) at these vertices.
# Or, simpler: use FFT on a grid of size 2N-1?
# If we sample f at the centers of the intervals? No, f is a step function.
# Let's stick to the FFT approach on a fine grid, it's robust and fast enough.
# FINE_N = 20000 is fast.
FINE_N = 20000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map fine grid points to interval indices
# Interval i covers [-1/4 + i*w, -1/4 + (i+1)*w)
# w = 0.5/N
W = 0.5 / N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT_FINE = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Metric: max(g) / sum(h^2) ? Or max(g) / (sum(h))^2 ?
    The task says "minimize the normalized maximum of its autoconvolution".
    Usually, for the first autocorrelation inequality, C1 is related to:
    max (f*f) / ||f||_2^2 ?
    Let's look at the seed program.
    Seed: return peak / l2_sq where l2_sq = sum(f*f).
    So the metric is max(autoconv) / L2_norm^2.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    # L2 norm squared
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return float('inf')
        
    # Expand h to fine grid
    f = h[CELL]
    
    # FFT convolution
    F = np.fft.rfft(f, N_FFT_FINE)
    conv = np.fft.irfft(F * F, N_FFT_FINE)[:2 * FINE_N - 1] * DX
    
    peak = np.max(conv)
    return float(peak / l2_sq)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for 150s limit
    
    # Precompute Fourier basis for low-frequency parameterization
    # We will parameterize h(t) = c0 + sum_{k=1}^K (a_k cos(4*pi*k*t) + b_k sin(4*pi*k*t))
    # t in [-1/4, 1/4]
    # This enforces global smoothness and eliminates high-frequency noise.
    K_MAX = 10  # Number of harmonics
    X_BASIS = np.linspace(-0.25, 0.25, N, endpoint=False)
    COS_BASIS = np.zeros((N, K_MAX))
    SIN_BASIS = np.zeros((N, K_MAX))
    for k in range(1, K_MAX + 1):
        COS_BASIS[:, k-1] = np.cos(4 * np.pi * k * X_BASIS)
        SIN_BASIS[:, k-1] = np.sin(4 * np.pi * k * X_BASIS)
    
    def fourier_to_h_fast(c0, a, b):
        h = c0 + a @ COS_BASIS.T + b @ SIN_BASIS.T
        return np.clip(h, 0.0, None)
    
    # Objective function for Fourier parameters
    def objective(params):
        c0 = params[0]
        a = params[1:1+K_MAX]
        b = params[1+K_MAX:1+2*K_MAX]
        h = fourier_to_h_fast(c0, a, b)
        if h.sum() < 1e-12:
            return 1e6
        return ratio_from_h(h)
    
    # Initial best: Uniform
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    # Phase 1: Fourier Domain Optimization
    # Use Nelder-Mead or L-BFGS-B on the Fourier coefficients.
    # L-BFGS-B is generally faster and more robust for this smooth landscape.
    
    rng = np.random.default_rng(42)
    
    # Try a few random initializations in Fourier space
    n_starts = 5
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
            
        # Random initialization
        # Start with a Gaussian-like profile?
        # A Gaussian in time is a Gaussian in frequency.
        # Let's just use small random perturbations around a constant.
        c0_init = rng.uniform(0.5, 1.0)
        a_init = np.zeros(K_MAX)
        b_init = np.zeros(K_MAX)
        for k in range(K_MAX):
            scale = 0.1 / (k + 1)
            a_init[k] = rng.normal(0, scale)
            b_init[k] = rng.normal(0, scale)
            
        x0 = np.concatenate([[c0_init], a_init, b_init])
        
        try:
            # Use L-BFGS-B
            res = opt.minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < best_val:
                h_opt = fourier_to_h_fast(res.x[0], res.x[1:1+K_MAX], res.x[1+K_MAX:1+2*K_MAX])
                v_opt = ratio_from_h(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt
        except Exception:
            pass
    
    # Phase 2: Local Refinement in Spatial Domain
    # The Fourier parameterization might be too restrictive for local features.
    # Use the best result from Phase 1 as a starting point for spatial local search.
    if time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(12345)
        current_h = best_h.copy()
        current_val = best_val
        
        # Simple local search with random perturbations
        for it in range(1000):
            if time.time() > deadline - 5.0:
                break
            
            # Random block perturbation
            mode = int(rng2.integers(0, 3))
            if mode == 0:
                # Additive bump
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(5, 30))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                amp = float(rng2.normal(0.0, 0.05))
                h_try = current_h.copy()
                xs = np.linspace(-1, 1, hi - lo)
                bump = np.exp(-xs**2)
                h_try[lo:hi] += amp * bump
                h_try = np.clip(h_try, 0.0, None)
            elif mode == 1:
                # Multiplicative scale
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(5, 30))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                f = float(rng2.uniform(0.9, 1.1))
                h_try = current_h.copy()
                h_try[lo:hi] *= f
                h_try = np.clip(h_try, 0.0, None)
            else:
                # Smoothing / Blurring
                h_try = current_h.copy()
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(3, 15))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                if hi - lo > 2:
                    h_try[lo:hi] = np.convolve(h_try[lo:hi], np.ones(3)/3, mode='same')
            
            v = ratio_from_h(h_try)
            if v < current_val:
                current_val = v
                current_h = h_try
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                # Reject
                pass
    
    return list(best_h)
