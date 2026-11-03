import numpy as np
import time

N = 600
DEADLINE = 130.0
START_TIME = time.time()

def _check_time():
    return time.time() - START_TIME < DEADLINE

def _autoconv_peak(h):
    """Compute the peak of the autoconvolution of h."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    h_pad = np.zeros(size)
    h_pad[:n] = h
    H = np.fft.fft(h_pad)
    conv = np.fft.ifft(H * H).real[:2*n-1]
    return np.max(conv)

def _project(h, target_sum=1.0):
    """Project onto {h >= 0, sum(h) = target_sum}."""
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s == 0:
        return np.full(len(h), target_sum / len(h))
    return h * (target_sum / s)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_h = None
    best_peak = float('inf')
    
    x = np.linspace(0, 1, N)
    
    # Generate diverse candidate functions
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N) / N)
    
    # Various Gaussians with different widths
    for sigma in [0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        g = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
        candidates.append(g / np.sum(g))
    
    # Triangular
    tri = np.maximum(0, 1 - 2 * np.abs(x - 0.5))
    candidates.append(tri / np.sum(tri))
    
    # Flat top with various edge widths
    for edge in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]:
        f = np.ones(N)
        ml = x < edge
        mr = x > 1 - edge
        f[ml] = x[ml] / edge
        f[mr] = (1 - x[mr]) / edge
        candidates.append(f / np.sum(f))
    
    # Cosine bump with various widths
    for width in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        c = np.zeros(N)
        mask = np.abs(x - 0.5) < width / 2
        c[mask] = 0.5 * (1 + np.cos(2 * np.pi * (x[mask] - 0.5) / width))
        candidates.append(c / np.sum(c))
    
    # Power law bumps
    for p in [0.5, 1, 1.5, 2, 3, 4]:
        pw = (1 - 2 * np.abs(x - 0.5)) ** p
        pw = np.maximum(pw, 0)
        candidates.append(pw / np.sum(pw))
    
    # Sinc-like
    for width in [0.3, 0.5, 0.7]:
        s = np.zeros(N)
        mask = np.abs(x - 0.5) < width / 2
        t = (x[mask] - 0.5) / (width / 2)
        s[mask] = np.sinc(t)
        s = np.maximum(s, 0)
        if np.sum(s) > 0:
            candidates.append(s / np.sum(s))
    
    # Evaluate all candidates
    for cand in candidates:
        if not _check_time():
            break
        peak = _autoconv_peak(cand)
        if peak < best_peak:
            best_peak = peak
            best_h = cand.copy()
    
    # Local optimization using coordinate descent
    if best_h is not None and _check_time():
        h = best_h.copy()
        current_peak = best_peak
        
        # Multi-pass coordinate descent with decreasing step sizes
        step_sizes = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        
        for step in step_sizes:
            if not _check_time():
                break
            for sweep in range(3):
                if not _check_time():
                    break
                for j in range(N):
                    if not _check_time():
                        break
                    old_val = h[j]
                    for delta in [step, -step]:
                        new_val = old_val + delta
                        if new_val < 0:
                            continue
                        h[j] = new_val
                        s = np.sum(h)
                        if s <= 0:
                            h[j] = old_val
                            continue
                        h_test = h / s
                        peak_test = _autoconv_peak(h_test)
                        if peak_test < current_peak:
                            current_peak = peak_test
                            best_peak = peak_test
                            best_h = h.copy()
                            break
                        else:
                            h[j] = old_val
    
    if best_h is None:
        best_h = np.ones(N) / N
    
    return list(best_h)
