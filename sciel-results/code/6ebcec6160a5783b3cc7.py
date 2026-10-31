"""Rewritten program: spectral-design / gradient-based optimization for the second autocorrelation inequality."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, ifft

N = 50
DEADLINE = 115.0  # safety margin before 120s kill

def _autoconvolution(h):
    """Compute the full autoconvolution (linear convolution) of h with itself.
    Returns array of length 2N-1."""
    # Use FFT for speed
    n_fft = 2 * N
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft)
    conv = conv.real
    return conv[:2 * N - 1]

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    All heights are nonnegative, so ||h||_1 = sum(h)."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    conv = _autoconvolution(h)
    sup = np.max(np.abs(conv))
    if sup <= 0:
        return 0.0
    return l2_sq / (l1 * sup)

def _negative_score(h):
    """Negative of the score for minimization."""
    return -_score(h)

def _project_nonnegative(h):
    """Project onto nonnegative orthant."""
    return np.maximum(h, 0.0)

def _sidon_init(seed):
    """Initialize with a Sidon-like set: choose positions to minimize autoconvolution peaks.
    Returns a sparse nonnegative vector."""
    rng = np.random.default_rng(seed)
    h = np.zeros(N)
    # Start with a few well-separated positions
    used = set()
    # Place N//3 + 1 points in a Sidon-like manner
    target_count = max(5, N // 3)
    attempts = 0
    while len(used) < target_count and attempts < 1000:
        pos = rng.integers(0, N)
        if pos not in used:
            # Check that differences are mostly unique (Sidon property)
            diffs = []
            for p in used:
                d = abs(pos - p)
                if d in diffs:
                    break
                diffs.append(d)
            else:
                used.add(pos)
        attempts += 1
    for pos in used:
        h[pos] = 1.0
    # Add some random small values
    extra = rng.random(N) * 0.3
    h += extra
    return h

def _random_init(seed):
    """Random nonnegative initialization."""
    rng = np.random.default_rng(seed)
    h = rng.random(N)
    # Normalize to have reasonable scale
    h = h / np.sum(h) * 5.0
    return h

def construct():
    start_time = time.time()
    best_h = None
    best_score = 0.0

    # Try multiple initializations and optimize each
    seeds = list(range(20))
    rng = np.random.default_rng(42)
    seeds += list(rng.integers(0, 10000, size=10))

    for seed in seeds:
        if time.time() - start_time > DEADLINE:
            break

        # Alternate between Sidon-like and random initializations
        if seed % 2 == 0:
            h0 = _sidon_init(seed)
        else:
            h0 = _random_init(seed)

        # Gradient-based optimization using L-BFGS-B
        try:
            res = minimize(
                _negative_score,
                h0,
                method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = _project_nonnegative(res.x)
            s = _score(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
        except Exception:
            pass

        # Also try a few local perturbations of the best found so far
        if best_h is not None and time.time() - start_time > DEADLINE - 30:
            break

    # If we have time, do a final round of local search around the best
    if best_h is not None and time.time() - start_time < DEADLINE - 10:
        rng = np.random.default_rng(12345)
        for _ in range(50):
            if time.time() - start_time > DEADLINE:
                break
            h_pert = best_h + rng.normal(0, 0.1, size=N)
            h_pert = _project_nonnegative(h_pert)
            try:
                res = minimize(
                    _negative_score,
                    h_pert,
                    method='L-BFGS-B',
                    bounds=[(0.0, None)] * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_opt = _project_nonnegative(res.x)
                s = _score(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
            except Exception:
                pass

    if best_h is None:
        # Fallback: uniform
        best_h = np.ones(N)

    # Ensure nonnegative
    best_h = np.maximum(best_h, 0.0)

    # Return as list of floats
    return [float(x) for x in best_h]
