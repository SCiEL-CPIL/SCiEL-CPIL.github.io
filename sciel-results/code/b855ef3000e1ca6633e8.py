"""Rewritten program: search for signed step function minimizing autoconvolution upper bound."""
N = 400
import numpy as np
import time

def _autoconv_bound(f):
    """Compute the L-infinity norm of the autoconvolution of the step function f.
    f is an array of N step heights. The step function is defined on [-1/4, 1/4]
    with N equal intervals. The autoconvolution is computed via FFT.
    """
    # The step function has value f[i] on interval [x_i, x_{i+1}] where
    # x_i = -1/4 + i * (1/2) / N
    # The autoconvolution (f*f)(x) = integral of f(t) f(x-t) dt
    # For step functions, this can be computed via the convolution of the
    # "density" which is f[i] / dx where dx = (1/2)/N
    # Actually, the autoconvolution integral: (f*f)(x) = ∫ f(t) f(x-t) dt
    # Since f is a step function, we can compute this via FFT on the sampled values.
    
    # The width of each interval
    dx = (1.0 / 2.0) / N
    
    # The autoconvolution at the grid points. For a step function with N intervals,
    # the autoconvolution has support on [-1/2, 1/2] and is a piecewise linear
    # function with 2N-1 linear segments. We can compute it via FFT.
    
    # Pad to length 2N for linear convolution
    n_fft = 2 * N
    f_pad = np.zeros(n_fft)
    f_pad[:N] = f
    
    # FFT-based convolution
    F = np.fft.fft(f_pad, n=n_fft)
    conv = np.fft.ifft(F * F).real
    
    # The convolution gives us the sum of products, but we need the integral.
    # The discrete convolution sum approximates the integral when multiplied by dx.
    # Actually, for the autoconvolution integral:
    # (f*f)(x) = ∫ f(t) f(x-t) dt
    # If we sample f at the midpoints of each interval, the Riemann sum is:
    # sum_i f(t_i) f(x - t_i) * dx
    # So the autoconvolution values are conv * dx (approximately)
    
    # The autoconvolution is defined on [-1/2, 1/2]. The FFT convolution gives
    # values at discrete points. The L-infinity norm is the max absolute value.
    autoconv = conv * dx
    
    # The autoconvolution should be symmetric and centered. Take the max absolute value.
    return np.max(np.abs(autoconv))


def _score(f):
    """Return the upper bound on C3 for the given step function."""
    return _autoconv_bound(f)


def construct():
    """Construct a signed step function with N intervals to minimize the autoconvolution upper bound."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial: all ones (seed)
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    # Try a few structured patterns first
    patterns = [
        np.ones(N),
        np.ones(N) * np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]),
        np.ones(N) * np.array([1.0 if i < N // 2 else -1.0 for i in range(N)]),
        np.ones(N) * np.array([1.0 if (i % 4 < 2) else -1.0 for i in range(N)]),
    ]
    
    for p in patterns:
        s = _score(p)
        if s < best_score:
            best_score = s
            best_f = p.copy()
    
    # Local search: flip individual signs to improve
    # Use a greedy approach with multiple passes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle order to avoid getting stuck
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > deadline:
                break
            # Try flipping sign at position i
            old_val = best_f[i]
            best_f[i] = -old_val
            new_score = _score(best_f)
            if new_score < best_score - 1e-12:
                best_score = new_score
                improved = True
            else:
                best_f[i] = old_val
    
    # Simulated annealing-like: try random flips with acceptance
    temp = 0.01
    while time.time() < deadline:
        # Pick a random position to flip
        i = np.random.randint(N)
        old_val = best_f[i]
        best_f[i] = -old_val
        new_score = _score(best_f)
        delta = new_score - best_score
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            best_score = new_score
        else:
            best_f[i] = old_val
        # Cool down
        temp *= 0.999
    
    return best_f
