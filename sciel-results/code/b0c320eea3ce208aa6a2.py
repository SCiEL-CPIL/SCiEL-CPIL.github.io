"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / s ** 2)


def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    idx = np.arange(N)

    best_h = np.ones(N)
    best_val = 2.0

    candidates = []

    for k in range(1, 20):
        for t in np.linspace(-0.9, 0.9, 20):
            candidates.append(1.0 + t * np.cos(2.0 * np.pi * k * idx / N))
    for k in range(1, 20):
        for t in np.linspace(-0.9, 0.9, 20):
            candidates.append(1.0 + t * np.sin(2.0 * np.pi * k * idx / N))
    for t in np.linspace(-0.9, 0.9, 20):
        candidates.append(1.0 + t * np.sin(2.0 * np.pi * idx ** 2 / N))
    for t in np.linspace(-0.9, 0.9, 20):
        candidates.append(1.0 + t * idx / (N - 1))
    for t in np.linspace(-0.9, 0.9, 20):
        x = idx / (N - 1)
        candidates.append(1.0 + t * (2.0 * x - 1.0) ** 2)

    for h in candidates:
        if time.time() > deadline - 50:
            break
        v = compute_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    if time.time() < deadline - 20:
        h0 = best_h.copy()
        s0 = np.sum(h0)
        h0 = h0 / s0

        def soft_max(x, beta=200.0):
            m = np.max(x)
            return m + np.log(np.sum(np.exp((x - m) * beta))) / beta

        def obj(x):
            conv = np.convolve(x, x)
            sm = soft_max(conv, beta=200.0)
            s = np.sum(x)
            return sm + 100.0 * (s - 1.0) ** 2

        try:
            res = minimize(obj, h0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 300})
            if time.time() < deadline:
                h_new = res.x
                v_new = compute_value(h_new)
                if v_new < best_val:
                    best_val = v_new
                    best_h = h_new.copy()
        except Exception:
            pass

    return best_h
# EVOLVE-BLOCK-END
