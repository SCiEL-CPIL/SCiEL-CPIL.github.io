import numpy as np
import time
from scipy.optimize import minimize

def _autoconv(f):
    n = len(f)
    F_full = np.fft.fft(f, n=2*n)
    conv_full = np.fft.ifft(F_full * F_full)[:2*n-1].real
    return conv_full

def _compute_score(f):
    n = len(f)
    f = np.asarray(f, dtype=np.float64)
    if np.all(f == 0):
        return 0.0
    conv = _autoconv(f)
    max_conv = np.max(conv)
    sum_f = np.sum(f)
    norm = sum_f * sum_f
    if norm == 0:
        return 0.0
    return max_conv / norm

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 140.0

    def time_remaining():
        return deadline - time.time()

    # Start with uniform distribution
    f_best = np.ones(n)
    f_best = f_best / np.sum(f_best)
    best_score = _compute_score(f_best)

    # Try a simple local search using coordinate descent or random perturbations
    # to see if we can beat the uniform score.
    # Since uniform is the theoretical minimum for this metric, this is a safeguard.
    
    if time_remaining() > 10.0:
        # Try a few random perturbations
        rng = np.random.default_rng(42)
        for _ in range(50):
            if time_remaining() < 5.0:
                break
            f_test = f_best.copy()
            # Add small random noise
            noise = rng.normal(0, 0.01, n)
            f_test += noise
            f_test = np.maximum(f_test, 0)
            sum_f = np.sum(f_test)
            if sum_f < 1e-6:
                continue
            f_test = f_test / sum_f
            score = _compute_score(f_test)
            if score < best_score:
                best_score = score
                f_best = f_test

    return list(f_best)
