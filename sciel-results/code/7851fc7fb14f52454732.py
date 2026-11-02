"""Optimized construction for the signed autocorrelation C3-prime task.
Combines continuous scipy optimization with discrete sign-flip refinement.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400


def _autoconv(x):
    """Full linear autoconvolution of x via FFT (zero-padded)."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(x, size)
    c = np.fft.irfft(F * np.conj(F), size)
    return c[:2 * n - 1]


def _score(x):
    """C3' = max|autoconv| / (integral)^2. Integral = sum(x)."""
    s = x.sum()
    if abs(s) < 1e-12:
        return np.inf
    c = _autoconv(x)
    return float(np.max(np.abs(c)) / (s * s))


def _score_and_grad(x, tau=0.1):
    """Smooth approximation of max|ac|/(sum)^2 with gradient via finite diff.
    Uses log-sum-exp style soft max for differentiability."""
    s = x.sum()
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(x)
    c = _autoconv(x)
    ac_abs = np.abs(c)
    # Soft max: tau * log(sum(exp(ac_abs/tau))) approximates max
    # But we want to minimize, so use the soft max value
    m = np.max(ac_abs)
    soft_max = m + tau * np.log(np.sum(np.exp((ac_abs - m) / tau)))
    val = soft_max / (s * s)
    
    # Numerical gradient (central differences on a few coordinates for speed)
    # Actually, let's just return the value; scipy can do its own numerical grad
    return val, None


def _continuous_optimize(x0, deadline, max_iter=500):
    """Use scipy to optimize continuously, then snap to signs."""
    x0 = x0.astype(np.float64)
    s0 = x0.sum()
    if abs(s0) < 1e-12:
        x0 = x0 + 0.1
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        return _score_and_grad(x)[0]
    
    # Bounds: keep values in [-2, 2] to avoid extreme values
    bounds = [(-2.0, 2.0)] * N
    
    try:
        res = minimize(
            objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': 2000, 'ftol': 1e-8}
        )
        if res.success or res.fun < _score(x0):
            return res.x
    except Exception:
        pass
    return x0


def _snap_to_signs(x):
    """Snap continuous values to ±1 while preserving relative magnitudes.
    Use threshold at 0, but scale to unit magnitude."""
    signs = np.sign(x)
    signs[signs == 0] = 1.0
    return signs


def _local_search(x, deadline, iters_per_step=5000):
    """Coordinate / sign-flip local search minimizing _score."""
    x = x.copy().astype(np.float64)
    best = _score(x)
    n = len(x)
    rng = np.random.default_rng(12345)
    
    while time.time() < deadline:
        # Try flipping a random subset of signs (block moves)
        block = rng.integers(1, max(2, n // 8))
        idx = rng.choice(n, size=block, replace=False)
        cand = x.copy()
        cand[idx] = -cand[idx]
        sc = _score(cand)
        if sc < best:
            best = sc
            x = cand
        
        # Single sign flips
        steps = min(iters_per_step, max(1000, int((deadline - time.time()) * 200)))
        for _ in range(steps):
            if time.time() >= deadline:
                break
            j = int(rng.integers(n))
            cand = x.copy()
            cand[j] = -cand[j]
            sc = _score(cand)
            if sc < best:
                best = sc
                x = cand
    
    return x, best


def _structured_seeds(rng):
    """Generate diverse structured seeds."""
    seeds = []
    
    # All ones (baseline)
    seeds.append(np.ones(N, dtype=np.float64))
    
    # Alternating
    seeds.append(np.tile([1.0, -1.0], N // 2).astype(np.float64))
    
    # Barker-like / low autocorrelation patterns
    for _ in range(8):
        x = rng.choice([-1.0, 1.0], size=N)
        # Add small DC offset
        x += 0.05
        seeds.append(x)
    
    # Block patterns with varying block sizes
    for block_size in [4, 8, 16, 32]:
        if N % block_size == 0:
            n_blocks = N // block_size
            # Random block signs
            block_signs = rng.choice([-1.0, 1.0], size=n_blocks)
            x = np.repeat(block_signs, block_size).astype(np.float64)
            seeds.append(x)
    
    # Smoothed random
    for _ in range(4):
        x = rng.choice([-1.0, 1.0], size=N)
        # Apply a light smoothing
        kernel = np.ones(3) / 3.0
        x_smooth = np.convolve(x, kernel, mode='same')
        x_smooth = np.sign(x_smooth)
        x_smooth[x_smooth == 0] = 1.0
        x_smooth += 0.05
        seeds.append(x_smooth.astype(np.float64))
    
    return seeds


def construct():
    """Return the sequence of signed step heights minimizing C3'."""
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    rng = np.random.default_rng(2024)
    
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)
    
    seeds = _structured_seeds(rng)
    
    for s in seeds:
        if time.time() >= deadline - 5:
            break
        
        s = s.astype(np.float64)
        # Ensure non-zero sum
        if abs(s.sum()) < 1e-12:
            s += 0.1
        
        # Phase 1: Continuous optimization
        x_cont = _continuous_optimize(s, deadline - 5, max_iter=200)
        
        # Phase 2: Snap to signs and refine
        x_signs = _snap_to_signs(x_cont)
        # Add small offset to avoid zero-sum issues during search
        x_search = x_signs + 0.01
        
        x_refined, sc = _local_search(x_search, deadline - 2)
        if sc < best_score:
            best_score = sc
            best_x = x_refined
    
    # Final refinement on the best found
    if time.time() < deadline - 3:
        x_final, sc_final = _local_search(best_x, deadline - 1, iters_per_step=10000)
        if sc_final < best_score:
            best_x = x_final
            best_score = sc_final
    
    return best_x
