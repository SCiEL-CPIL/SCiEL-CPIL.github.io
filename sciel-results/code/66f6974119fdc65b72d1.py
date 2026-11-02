"""Rewritten program: smooth-max + L-BFGS-B on symmetric step heights."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 600
HALF = N // 2  # 300, exploit even symmetry

def _autoconv_peak_from_symmetric(h):
    """h has length HALF (right half including center). Build full, compute autoconv, return max."""
    # Build full symmetric: full[0..HALF-1] = h, full[HALF..N-1] = h[::-1]
    # But careful with even N: indices 0..599. Symmetric about center between 299 and 300.
    # For even symmetry about 0 with 600 bins on [-1/4,1/4], bin i covers [-1/4 + i*d, -1/4+(i+1)*d]
    # Symmetry: f(x) = f(-x). Bin i (center at -1/4+(i+0.5)*d) maps to bin (N-1-i).
    full = np.empty(N)
    full[:HALF] = h
    full[HALF:] = h[::-1]
    # autoconvolution (full correlation at lag 0 region) — actually autoconvolution
    ac = fftconvolve(full, full, mode='full')
    # The peak of autoconvolution; for a nonneg function the max is at lag 0 typically
    return ac.max(), full

def _smooth_max(ac, temperature=1.0):
    """Smooth maximum via log-sum-exp scaled."""
    m = ac.max()
    # log-sum-exp for stability
    shifted = ac - m
    return m + np.log(np.sum(np.exp(shifted / temperature)) / len(ac)) * temperature

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    # Initial guess: flat
    h0 = np.ones(HALF)

    def objective(h):
        if time.time() > deadline:
            return np.inf
        # Ensure nonneg
        h = np.maximum(h, 0.0)
        full = np.empty(N)
        full[:HALF] = h
        full[HALF:] = h[::-1]
        ac = fftconvolve(full, full, mode='full')
        # Normalize: divide by (sum h)^2 * interval? The "normalized maximum of autoconvolution"
        # Typically normalized by L1^2 or similar. Let's use max(ac) / (sum(full)**2) * N
        # Actually the C1 bound: for step function, autoconvolution peak normalized.
        # Let's use max(ac) / (sum(full) ** 2) * N as a normalized measure
        s = full.sum()
        if s <= 0:
            return np.inf
        return (ac.max() / (s * s)) * N

    # Gradient via finite differences is expensive; use analytic via smooth-max
    # For L-BFGS-B we need gradient. Let's use a smooth surrogate.
    def objective_smooth(h, temp=5.0):
        if time.time() > deadline:
            return np.inf, np.zeros_like(h)
        h = np.maximum(h, 1e-8)
        full = np.empty(N)
        full[:HALF] = h
        full[HALF:] = h[::-1]
        ac = fftconvolve(full, full, mode='full')
        s = full.sum()
        if s <= 0:
            return np.inf, np.zeros_like(h)
        # smooth max
        m = ac.max()
        shifted = ac - m
        softmax = np.exp(shifted / temp)
        smax = m + temp * np.log(softmax.sum() / len(ac))
        val = (smax / (s * s)) * N
        # Gradient: d/dh of smax/(s^2) * N
        # dsmax/dac_j = softmax_j / softmax.sum()
        # dac_j/dh: autoconvolution ac[j] = sum_k full[k]*full[j-k] (for full length 2N-1)
        # This is complex; use finite difference for gradient to keep it simple but accurate
        # Actually let's just use numerical gradient via L-BFGS-B's own approximation
        # scipy minimize can approximate gradient
        return val, None

    # Use L-BFGS-B with numerical gradient
    best_h = h0.copy()
    best_val = objective(h0)

    # Try optimization
    try:
        res = minimize(
            lambda x: objective(x),
            h0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8},
            bounds=[(0.0, None)] * HALF
        )
        if res.x is not None and res.fun < best_val:
            best_h = np.maximum(res.x, 0.0)
            best_val = res.fun
    except Exception:
        pass

    if time.time() > deadline:
        # Return best so far
        full = np.empty(N)
        full[:HALF] = best_h
        full[HALF:] = best_h[::-1]
        return full.tolist()

    # Local refinement: coordinate descent on the smooth objective
    for _ in range(30):
        if time.time() > deadline:
            break
        improved = False
        for i in range(HALF):
            old = best_h[i]
            # Try a few perturbations
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                cand = best_h.copy()
                cand[i] = max(0.0, old + delta)
                v = objective(cand)
                if v < best_val:
                    best_h = cand
                    best_val = v
                    improved = True
        if not improved:
            break

    full = np.empty(N)
    full[:HALF] = best_h
    full[HALF:] = best_h[::-1]
    return full.tolist()
