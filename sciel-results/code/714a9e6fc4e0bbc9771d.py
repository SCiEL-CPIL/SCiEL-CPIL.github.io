"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


def _score(h):
    """Compute max |autoconv| / (sum(h))^2. Lower is better."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    n = len(h)
    size = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)


def _build_from_params(params, n=N, n_kernels=6):
    """
    Build a signal from parametric representation:
    sum of n_kernels shifted, scaled, signed rectangular pulses.
    params layout: for each kernel k (0..n_kernels-1):
        params[3k+0] = center position (0..n)
        params[3k+1] = width (1..n)
        params[3k+2] = amplitude (can be negative)
    The signal is the sum of these rectangular pulses, clipped to [0, n-1].
    """
    h = np.zeros(n)
    for k in range(n_kernels):
        center = params[3*k + 0]
        width = abs(params[3*k + 1]) + 1.0
        amp = params[3*k + 2]
        lo = int(np.floor(center - width / 2.0))
        hi = int(np.ceil(center + width / 2.0))
        lo = max(0, lo)
        hi = min(n, hi)
        if hi > lo:
            h[lo:hi] += amp
    return h


def _parametric_score(params):
    """Score function for differential evolution."""
    h = _build_from_params(params)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    n = len(h)
    size = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)


def _local_refine(h, time_budget=20.0):
    """
    Local refinement: iteratively adjust individual elements to reduce score.
    Uses FFT-based scoring.
    """
    t0 = time.time()
    current = _score(h)
    n = len(h)
    size = 1 << (2 * n - 1).bit_length()
    
    # Precompute FFT size
    improved = True
    while improved and (time.time() - t0) < time_budget:
        improved = False
        # Try random subset of positions
        rng_local = np.random.default_rng(int((time.time() - t0) * 1000) % (2**31))
        for _ in range(200):
            if (time.time() - t0) >= time_budget:
                break
            i = rng_local.integers(0, n)
            orig = h[i]
            # Try several deltas
            for delta in [0.1, -0.1, 0.3, -0.3, 0.5, -0.5]:
                h[i] = orig + delta
                s = np.sum(h)
                if abs(s) < 1e-12:
                    h[i] = orig
                    continue
                H = np.fft.rfft(h, size)
                ac = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
                max_abs = np.max(np.abs(ac))
                sc = max_abs / (s * s)
                if sc < current - 1e-10:
                    current = sc
                    improved = True
                    break
            h[i] = orig
    return h, current


def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin
    
    # Baseline
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Parametric approach: differential evolution on block-pulse parameters
    n_kernels = 5
    n_params = 3 * n_kernels
    
    # Bounds: center in [0, N], width in [1, N], amplitude in [-3, 3]
    bounds = []
    for k in range(n_kernels):
        bounds.append((0.0, float(N)))       # center
        bounds.append((1.0, float(N)))       # width
        bounds.append((-3.0, 3.0))           # amplitude
    
    # Initial population hint: start near "all ones" which is one pulse of width N, amp 1
    # But DE will explore. Let's give it a reasonable start.
    
    # Time budget for DE
    de_time = 60.0
    max_iter = 80
    popsize = 15
    
    try:
        result = differential_evolution(
            _parametric_score,
            bounds,
            maxiter=max_iter,
            popsize=popsize,
            tol=1e-6,
            seed=42,
            polish=False,
            init='sobol',
            update_interval=1
        )
        
        if result.success or result.fun < best_score:
            h_de = _build_from_params(result.x)
            sc_de = _score(h_de)
            if sc_de < best_score:
                best_score = sc_de
                best_h = h_de
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return best_h
    
    # Local refinement
    remaining = deadline - time.time() - 5.0
    if remaining > 5:
        best_h, best_score = _local_refine(best_h.copy(), time_budget=remaining)
    
    # One more round: try a few structured patterns and refine
    if time.time() < deadline - 10:
        patterns = [
            np.ones(N),
            np.ones(N) * (1.0 + 0.05 * np.sin(2 * np.pi * np.arange(N) / N)),
            np.clip(1.0 + 0.2 * np.cos(2 * np.pi * np.arange(N) / N), 0.5, 1.5),
        ]
        for pat in patterns:
            if time.time() > deadline - 10:
                break
            sc = _score(pat)
            if sc < best_score:
                best_score = sc
                best_h = pat.copy()
        
        if time.time() < deadline - 5:
            rem = deadline - time.time() - 3.0
            if rem > 3:
                best_h, best_score = _local_refine(best_h.copy(), time_budget=rem)
    
    return best_h

# EVOLVE-BLOCK-END
