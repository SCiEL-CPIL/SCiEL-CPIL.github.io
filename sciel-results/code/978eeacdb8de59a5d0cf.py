"""Optimized program for the Third Autocorrelation Inequality task.

Strategy: Use a spectral / Fourier-domain approach to minimize the peak of
the autoconvolution. The key insight is that the autoconvolution in the
time domain corresponds to the squared magnitude of the DFT in the frequency
domain. By working in the frequency domain, we can more directly control
the peak of the autoconvolution.

Additionally, we use a combination of:
1. Randomized spectral initialization
2. Nelder-Mead optimization in the time domain with a smooth-max surrogate
3. Multiple restarts with different random seeds
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 400


def compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    idx = np.arange(N)

    best_h = np.ones(N)
    best_val = 2.0

    def try_candidate(h):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        if not np.all(np.isfinite(h)):
            return
        v = compute_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Phase 1: Spectral-based initialization
    # The autoconvolution c[k] = sum_i h[i] * h[i+k]
    # In frequency domain: C = |H|^2 where H = FFT(h)
    # We want to minimize max(c), which is related to the peak of |H|^2
    
    # Try various spectral profiles
    for trial in range(20):
        if time.time() > deadline - 60:
            break
        # Create a random spectral mask
        freqs = np.fft.rfftfreq(N)
        # Try different bandwidths and center frequencies
        bandwidth = np.random.uniform(0.05, 0.4)
        center = np.random.uniform(-0.3, 0.3)
        
        # Gaussian spectral profile
        H = np.exp(-((freqs - center) ** 2) / (2 * bandwidth ** 2))
        # Add some randomness
        H *= (1.0 + 0.3 * np.random.randn(len(H)))
        
        # Convert back to time domain
        h = np.fft.irfft(H, n=N)
        
        # Normalize so sum is 1
        s = np.sum(h)
        if abs(s) > 1e-10:
            h = h / s
        try_candidate(h)

    # Phase 2: Parametric search over diverse structures
    candidates = []

    # Cosine modes
    for k in range(1, 20):
        for t in np.linspace(-1.5, 1.5, 15):
            candidates.append(1.0 + t * np.cos(2.0 * np.pi * k * idx / N))
    # Sine modes
    for k in range(1, 20):
        for t in np.linspace(-1.5, 1.5, 15):
            candidates.append(1.0 + t * np.sin(2.0 * np.pi * k * idx / N))
    # Chirp signals
    for mu in [0.001, 0.005, 0.01, 0.05, 0.1]:
        for t in np.linspace(-1.0, 1.0, 10):
            phase = 2.0 * np.pi * (idx / N + mu * (idx / N) ** 2)
            candidates.append(1.0 + t * np.cos(phase))
    # Two-frequency mix
    for k1 in [1, 2, 3, 5, 7]:
        for k2 in [2, 3, 4, 6, 8, 10]:
            for t1 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                for t2 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    candidates.append(1.0 + t1 * np.cos(2.0 * np.pi * k1 * idx / N) + t2 * np.sin(2.0 * np.pi * k2 * idx / N))

    for h in candidates:
        if time.time() > deadline - 40:
            break
        s = np.sum(h)
        if abs(s) > 1e-10:
            try_candidate(h / s)

    # Phase 3: Local refinement using soft-max surrogate with multiple restarts
    if time.time() < deadline - 20:
        def soft_max(x, beta=300.0):
            m = np.max(x)
            return m + np.log(np.sum(np.exp((x - m) * beta))) / beta

        def make_obj(beta):
            def obj(x):
                conv = np.convolve(x, x)
                sm = soft_max(conv, beta=beta)
                s = np.sum(x)
                penalty = 500.0 * (s - 1.0) ** 2
                return sm + penalty
            return obj

        # Try multiple optimization methods and starting points
        start_points = [best_h.copy() / np.sum(best_h)]
        
        # Add perturbed starting points
        for _ in range(8):
            if time.time() > deadline - 15:
                break
            pert = start_points[0].copy() + 0.1 * np.random.randn(N)
            s_p = np.sum(pert)
            if abs(s_p) > 1e-10:
                start_points.append(pert / s_p)

        for sp in start_points:
            if time.time() > deadline - 10:
                break
            for beta in [200.0, 400.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    obj = make_obj(beta)
                    res = minimize(obj, sp, method='Nelder-Mead', 
                                 options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
                    if time.time() > deadline - 3:
                        break
                    h_new = res.x
                    s_new = np.sum(h_new)
                    if abs(s_new) > 1e-10:
                        v_new = compute_value(h_new)
                        if v_new < best_val:
                            best_val = v_new
                            best_h = h_new.copy()
                            start_points[0] = h_new.copy() / np.sum(h_new)
                except Exception:
                    pass

    # Phase 4: Additional refinement with different beta values
    if time.time() < deadline - 8:
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) > 1e-10:
            h0 = h0 / s0

        for beta in [100.0, 150.0, 250.0, 500.0]:
            if time.time() > deadline - 3:
                break

            def obj_b(x, b=beta):
                conv = np.convolve(x, x)
                m = np.max(conv)
                sm = m + np.log(np.sum(np.exp((conv - m) * b))) / b
                s = np.sum(x)
                return sm + 500.0 * (s - 1.0) ** 2

            try:
                res = minimize(obj_b, h0, method='L-BFGS-B',
                             options={'maxiter': 200, 'maxfun': 600, 'gtol': 1e-7})
                if time.time() > deadline - 2:
                    break
                h_new = res.x
                v_new = compute_value(h_new)
                if v_new < best_val:
                    best_val = v_new
                    best_h = h_new.copy()
                    h0 = h_new.copy() / np.sum(h_new)
            except Exception:
                pass

    return best_h
