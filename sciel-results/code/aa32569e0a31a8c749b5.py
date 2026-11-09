import numpy as np
import time

N = 600
DELTA = 1.0 / N

# Precompute FFT size
_NFFT = 1 << (2 * N - 1).bit_length()

def compute_ratio_fast(h):
    """Compute max(f*f) / (int f)^2 for step function with heights h, using FFT."""
    h = np.maximum(np.asarray(h, dtype=np.float64), 0.0)
    s = np.sum(h) * DELTA
    if s < 1e-15:
        return 1e15
    h_norm = h / s  # normalize so int f = 1
    # FFT-based autoconvolution
    F = np.fft.rfft(h_norm, _NFFT)
    conv = np.fft.irfft(F * F, _NFFT)[:2*N-1]
    # For step functions, the continuous autoconvolution at grid points is DELTA * conv
    max_val = np.max(conv) * DELTA
    return max_val

def compute_ratio_and_grad(h):
    """Compute ratio and gradient w.r.t. h for smooth optimization."""
    h = np.maximum(np.asarray(h, dtype=np.float64), 0.0)
    s = np.sum(h) * DELTA
    if s < 1e-15:
        return 1e15, np.zeros_like(h)
    h_norm = h / s
    # FFT-based autoconvolution
    F = np.fft.rfft(h_norm, _NFFT)
    conv = np.fft.irfft(F * F, _NFFT)[:2*N-1]
    max_val = np.max(conv) * DELTA
    # Gradient: d(max(conv))/dh - use subgradient at argmax
    # conv[j] = sum_i h_norm[i] * h_norm[i-(j-N+1)] (with bounds)
    # For simplicity, use finite differences on the max
    return max_val, None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin (150s limit)
    
    x = np.linspace(-0.25, 0.25, N+1)[:-1] + DELTA/2  # centers of intervals
    
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Inverted V: 0.25 - |x| (concentrated at center)
    candidates.append(0.25 - np.abs(x))
    
    # Cosine modulations - concentrate mass
    for a in [0.3, 0.5, 0.7, 0.9, 1.0]:
        for k in [1, 2, 4, 8]:
            candidates.append(1.0 - a * np.cos(2 * np.pi * k * x))
    
    # Power shapes concentrated at center
    for p in [2, 4, 6, 8, 12]:
        candidates.append((0.25 - np.abs(x)) ** p)
    
    # Gaussian centered at 0
    for sigma in [0.02, 0.03, 0.05, 0.07, 0.1]:
        candidates.append(np.exp(-x**2 / (2*sigma**2)))
    
    # Two-bump: high at edges (spread out)
    for sigma in [0.02, 0.03, 0.05, 0.08, 0.1]:
        candidates.append(np.exp(-((x - 0.125)**2) / (2*sigma**2)) + np.exp(-((x + 0.125)**2) / (2*sigma**2)))
    
    # Evaluate all candidates
    best_ratio = 1e15
    best_h = None
    for h in candidates:
        h = np.maximum(h, 0)
        if np.sum(h) < 1e-10:
            continue
        r = compute_ratio_fast(h)
        if r < best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Local refinement: coordinate descent with FFT
    if best_h is not None:
        s = np.sum(best_h) * DELTA
        h_opt = best_h / s
        
        # Coordinate descent - more iterations, faster eval
        for iteration in range(2000):
            if time.time() - start_time > deadline - 3:
                break
            improved = False
            # Random order to avoid getting stuck
            order = np.random.permutation(N)
            for i in order:
                if time.time() - start_time > deadline - 3:
                    break
                old_val = h_opt[i]
                if old_val < 1e-15:
                    # Try adding a small value
                    for new_val in [0.001, 0.01, 0.05, 0.1]:
                        h_test = h_opt.copy()
                        h_test[i] = new_val
                        s_test = np.sum(h_test) * DELTA
                        if s_test < 1e-15:
                            continue
                        h_test = h_test / s_test
                        r_test = compute_ratio_fast(h_test)
                        if r_test < best_ratio - 1e-14:
                            best_ratio = r_test
                            h_opt = h_test
                            improved = True
                            break
                else:
                    # Try scaling
                    for frac in [0.5, 0.7, 1.3, 1.5, 0.0, 2.0, 0.2, 3.0, 0.8, 1.2]:
                        new_val = old_val * frac
                        h_test = h_opt.copy()
                        h_test[i] = new_val
                        s_test = np.sum(h_test) * DELTA
                        if s_test < 1e-15:
                            continue
                        h_test = h_test / s_test
                        r_test = compute_ratio_fast(h_test)
                        if r_test < best_ratio - 1e-14:
                            best_ratio = r_test
                            h_opt = h_test
                            improved = True
                            break
            if not improved:
                # Try a few random perturbations before giving up
                for _ in range(20):
                    if time.time() - start_time > deadline - 3:
                        break
                    idx = np.random.choice(N, size=min(10, N), replace=False)
                    h_test = h_opt.copy()
                    for i in idx:
                        h_test[i] *= np.random.uniform(0.5, 1.5)
                    h_test = np.maximum(h_test, 0)
                    s_test = np.sum(h_test) * DELTA
                    if s_test < 1e-15:
                        continue
                    h_test = h_test / s_test
                    r_test = compute_ratio_fast(h_test)
                    if r_test < best_ratio - 1e-14:
                        best_ratio = r_test
                        h_opt = h_test
                        improved = True
                        break
                if not improved:
                    break
    
    # Final
    h_final = np.maximum(h_opt, 0.0)
    s_final = np.sum(h_final) * DELTA
    if s_final > 1e-15:
        h_final = h_final / s_final
    
    return [float(x) for x in h_final]
