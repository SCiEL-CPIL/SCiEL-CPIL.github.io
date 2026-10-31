"""Minimize normalized max autoconvolution via direct gradient-based search.

Structural change: instead of a 1-parameter cosine-power family, we directly
optimize the 600 nonnegative heights using projected gradient descent on the
objective  F(h) = max_j c[j] / (sum h)^2  where c = convolve(h,h,'full').
This is a different modelling family (full-dimensional direct search vs.
low-dimensional parametric family) and a different method (first-order
gradient descent with projection onto nonnegativity vs. scalar bounded
minimization).
"""

import time
import numpy as np

N = 600
DELTA = 0.5 / N
DEADLINE = 115.0  # safety margin before 150s kill


def _objective(h):
    """max_j c[j] / (sum h)^2  -- the natural C1-type normalized peak."""
    s = float(np.sum(h))
    if s <= 0:
        return float('inf')
    c = np.convolve(h, h, mode='full')
    return float(np.max(c)) / (s * s)


def _grad_objective(h):
    """Gradient of F(h) = max_j c[j] / S^2  where S = sum(h).

    Let k = argmax_j c[j].  Then dF/dh_i = (2 * c[k]_i / S^2) - 2 * F * h_i / S
    where c[k]_i = sum_j h_j h_{j-(k-i)}  (the i-th element of the k-th
    cross-correlation slice).  We compute this via FFT for speed.
    """
    s = float(np.sum(h))
    if s <= 0:
        return np.zeros_like(h), float('inf')
    
    # Compute full autoconvolution via FFT
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    # c[m] for m in 0..2N-2 maps to C[m] for m in 0..N-1, and C[2N-1-m] for m in N..2N-2
    # Actually np.convolve(h,h,'full') has length 2N-1.
    # C from irfft of H*H gives circular conv of length n_fft.
    # For linear conv: c_linear[m] = C[m] for m < N, and c_linear[m] = C[m - n_fft] for m >= N
    # But simpler: just use np.convolve for the argmax, then compute gradient analytically.
    
    c_full = np.convolve(h, h, mode='full')  # length 2N-1
    k = int(np.argmax(c_full))
    peak = float(c_full[k])
    F = peak / (s * s)
    
    # d(c_k)/d(h_i) = sum_j h_j * d/d(h_i)[h_{j-k+i}]... 
    # Actually c_k = sum_i h_i * h_{i-k+N-1} if we index properly.
    # c_full[m] = sum_i h[i] * h[i+m-(N-1)]  ... let's think differently.
    # c_full[m] = sum_{i} h[i] * h[i - m + (N-1)]  for valid indices
    # Equivalently: c_full[m] = sum_{j} h[j] * h[j + m - (N-1)]
    # d c_full[m] / d h[i] = h[i + m - (N-1)] + h[i - m + (N-1)]  (if indices valid)
    # This is the auto-correlation: (h * h_flipped)[i] shifted.
    
    # The gradient of c_k w.r.t. h is: h shifted by (k - (N-1)) plus h shifted by ((N-1) - k)
    # More carefully: c_k = sum_j h[j] h[j + k - N + 1]
    # d c_k / d h[i] = h[i + k - N + 1] + h[i - k + N - 1]  (with h out-of-range = 0)
    
    shift = k - (N - 1)
    # d c_k / d h[i] = h[i + shift] + h[i - shift]  (circular? no, zero-padded)
    dc = np.zeros(N)
    for i in range(N):
        j1 = i + shift
        j2 = i - shift
        if 0 <= j1 < N:
            dc[i] += h[j1]
        if 0 <= j2 < N:
            dc[i] += h[j2]
    
    # F = c_k / S^2
    # dF/dh_i = (dc_k/dh_i * S^2 - c_k * 2*S * 1) / S^4 = dc_k/dh_i / S^2 - 2*c_k/(S^3)
    grad = dc / (s * s) - 2.0 * peak / (s * s * s)
    
    return grad, F


def construct():
    global _t0
    _t0 = time.time()
    
    # Start from uniform (the seed)
    h = np.ones(N)
    best_h = h.copy()
    best_val = _objective(h)
    
    # Try a few different initializations
    inits = [
        np.ones(N),
        np.cos(np.pi * np.linspace(0, 1, N)) ** 2,  # raised cosine
        np.exp(-2.0 * (np.linspace(-1, 1, N)) ** 2),  # Gaussian
        np.clip(1.0 - 0.5 * np.abs(np.linspace(-1, 1, N)), 0.1, None),  # tent
    ]
    
    # Evaluate all inits, keep best
    for init in inits:
        init = np.clip(init, 0.0, None)
        v = _objective(init)
        if v < best_val:
            best_val = v
            best_h = init.copy()
    
    h = best_h.copy()
    
    # Projected gradient descent
    lr = 0.01
    for step in range(5000):
        if time.time() - _t0 > DEADLINE:
            break
        
        grad, F = _grad_objective(h)
        if not np.isfinite(F):
            break
        
        h_new = h - lr * grad
        h_new = np.clip(h_new, 0.0, None)  # project to nonnegative
        
        # Normalize to keep sum constant (scale invariance of objective)
        s = np.sum(h_new)
        if s > 0:
            h_new = h_new / s * np.sum(h)
        
        v = _objective(h_new)
        if v < best_val:
            best_val = v
            best_h = h_new.copy()
            h = h_new
            # Adaptive learning rate
            if step % 500 == 0 and step > 0:
                lr = max(lr * 0.8, 1e-5)
        else:
            h = h_new
            lr = max(lr * 0.95, 1e-6)
    
    # Final safety: ensure we don't return something worse than uniform
    uniform_val = _objective(np.ones(N))
    if best_val > uniform_val + 1e-12:
        best_h = np.ones(N)
    
    return [float(v) for v in best_h]


_t0 = time.time()
