"""
Autocorrelation inequality optimizer using FFT-based evaluation,
coordinate descent with smart initialization, and Nelder-Mead polish
on a reduced parameterization.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 135  # wall-clock seconds; keep margin before 150s kill


def _autoconvolve_fft(h):
    """Compute full autoconvolution using FFT (faster than np.convolve for large N)."""
    size = 2 * N - 1
    fsize = 1 << (size - 1).bit_length()
    F = np.fft.rfft(h, fsize)
    conv = np.fft.irfft(F * F, fsize)[:size]
    return conv


def _evaluate(h):
    """Return (value, valid) for the exact task objective."""
    if not np.all(np.isfinite(h)):
        return np.inf, False
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf, False
    conv = _autoconvolve_fft(h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return np.inf, False
    return val, True


def _normalize_sum(h, target=1.0):
    """Scale h so that sum(h) = target, if possible."""
    s = float(np.sum(h))
    if abs(s) < 1e-15:
        return h
    return h * (target / s)


def _init_spread_signs(rng):
    """Initialize with spread magnitudes and structured signs to reduce peak autocorrelation."""
    h = np.zeros(N)
    # Use a base pattern: alternating blocks with decreasing amplitude
    # This reduces constructive interference in autoconvolution
    block_size = 10
    for i in range(N):
        block = i // block_size
        # Alternating sign per block, with smooth amplitude decay
        sign = 1.0 if block % 2 == 0 else -1.0
        amp = 1.0 / (1.0 + 0.1 * block)
        h[i] = sign * amp * rng.uniform(0.8, 1.2)
    # Ensure sum is positive by flipping if needed
    if np.sum(h) < 0:
        h = -h
    return h


def _init_two_level(rng):
    """Two-level initialization: values are either +a or -b, with structure."""
    h = np.zeros(N)
    # Random assignment of +a and -b with ~50/50 split
    mask = rng.random(N) < 0.5
    a = 1.0
    b = 0.5
    h[mask] = a
    h[~mask] = -b
    # Add small random perturbation
    h += rng.normal(0, 0.01, N)
    if np.sum(h) < 0:
        h = -h
    return h


def _init_block_pattern(rng):
    """Block-structured initialization: groups of same-sign values."""
    h = np.zeros(N)
    n_blocks = rng.randint(20, 80)
    block_len = N // n_blocks
    sign = 1.0
    for b in range(n_blocks):
        start = b * block_len
        end = min(start + block_len + rng.randint(0, 3), N)
        amp = rng.uniform(0.5, 1.5)
        h[start:end] = sign * amp
        sign *= -1
    h += rng.normal(0, 0.005, N)
    if np.sum(h) < 0:
        h = -h
    return h


def _coordinate_descent(h, max_iters=2000, tol=1e-10, deadline=None):
    """
    Coordinate descent: for each coordinate i, find the optimal value
    that minimizes the objective, holding others fixed.
    Uses FFT for fast evaluation.
    """
    current_val, valid = _evaluate(h)
    if not valid:
        return h, np.inf
    
    h = h.copy()
    n_coord = len(h)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n_coord):
            if deadline is not None and time.time() > deadline:
                break
            
            # Save original value
            orig = h[i]
            
            # Try a few candidate values around current
            # The objective as a function of h[i] is piecewise quadratic-ish
            # due to max of autoconvolution terms
            candidates = [orig]
            # Perturbations
            for delta in [-0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5]:
                candidates.append(orig + delta)
            # Also try setting to 0
            candidates.append(0.0)
            # Try scaling
            if abs(orig) > 1e-10:
                candidates.append(orig * 0.5)
                candidates.append(orig * 2.0)
            
            best_cand = orig
            best_val = current_val
            
            for c in candidates:
                h[i] = c
                val, valid = _evaluate(h)
                if valid and val < best_val:
                    best_val = val
                    best_cand = c
            
            h[i] = best_cand
            if best_cand != orig:
                improved = True
                current_val = best_val
        
        if not improved:
            break
    
    return h, current_val


def _nelder_mead_reduced(h, deadline):
    """
    Use Nelder-Mead on a reduced set of parameters:
    We fix the structure and optimize a few global parameters.
    """
    # Extract: mean, std, and a few key values
    mean = float(np.mean(h))
    std = float(np.std(h))
    
    def objective(params):
        m, s = params
        h_new = mean + (h - mean) * (s / (std + 1e-15))
        val, valid = _evaluate(h_new)
        return val if valid else 1e10
    
    x0 = [mean, std]
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success or res.fun < _evaluate(h)[0]:
            m, s = res.x
            h_new = mean + (h - mean) * (s / (std + 1e-15))
            return h_new
    except Exception:
        pass
    return h


def construct():
    start = time.time()
    
    def _elapsed():
        return time.time() - start
    
    # Generate multiple diverse initializations
    rng = np.random.RandomState(42)
    candidates = []
    
    # 1. Spread signs initialization
    for trial in range(5):
        candidates.append(_init_spread_signs(rng))
    
    # 2. Two-level initialization
    for trial in range(5):
        candidates.append(_init_two_level(rng))
    
    # 3. Block pattern
    for trial in range(5):
        candidates.append(_init_block_pattern(rng))
    
    # 4. The known good spike candidate (from current program)
    h_spike = np.zeros(N)
    h_spike[0] = 0.5
    h_spike[1] = 0.5
    h_spike[2:200] = -0.001
    candidates.append(h_spike)
    
    # 5. All-ones (seed)
    candidates.append(np.ones(N))
    
    # 6. Random with positive bias
    for trial in range(5):
        h = rng.normal(1.0, 0.5, N)
        candidates.append(h)
    
    # Evaluate all candidates, keep the best few
    scored = []
    for h in candidates:
        val, valid = _evaluate(h)
        if valid:
            scored.append((val, h))
    
    if not scored:
        # Fallback
        h = np.ones(N)
        return h
    
    scored.sort(key=lambda x: x[0])
    best_val, best_h = scored[0]
    
    # Coordinate descent on the top candidates
    deadline_cd = start + DEADLINE - 20
    for i, (val, h) in enumerate(scored[:5]):
        if _elapsed() > DEADLINE - 25:
            break
        if val >= best_val and i > 0:
            break
        
        h_opt, val_opt = _coordinate_descent(h, max_iters=500, deadline=deadline_cd)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Nelder-Mead polish on reduced parameters
    if _elapsed() < DEADLINE - 10:
        h_nm = _nelder_mead_reduced(best_h, deadline=start + DEADLINE - 5)
        val_nm, valid = _evaluate(h_nm)
        if valid and val_nm < best_val:
            best_val = val_nm
            best_h = h_nm
    
    # Final local random search
    if _elapsed() < DEADLINE - 5:
        rng2 = np.random.RandomState(123)
        for _ in range(100):
            if _elapsed() > DEADLINE - 5:
                break
            h_try = best_h + rng2.normal(0, 0.02, N)
            val_try, valid = _evaluate(h_try)
            if valid and val_try < best_val:
                best_val = val_try
                best_h = h_try
    
    # Final validity guard
    val_final, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val_final):
        best_h = np.ones(N)
    
    return best_h
