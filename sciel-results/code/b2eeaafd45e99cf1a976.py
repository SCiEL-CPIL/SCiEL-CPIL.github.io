"""Optimized program for third autocorrelation inequality."""
import numpy as np
import time

N = 400
_DEADLINE = 140  # seconds, safety margin under 150s

def _autocorr_coeffs(h):
    """Compute full autoconvolution coefficients via FFT for speed."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size=size)
    ac = np.fft.irfft(H * H, size=size)
    return ac[:2*n-1]

def _objective(h):
    """Compute the raw value."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    ac = _autocorr_coeffs(h)
    return abs(2.0 * N * np.max(ac) / (s * s))

def _smooth_max(ac, beta):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(ac)
    if beta * m < 500:
        return m + np.log(np.sum(np.exp(beta * (ac - m)))) / beta
    return m

def _construct_initial():
    """Generate a diverse initial candidate."""
    # Start with a configuration that has low peak autocorrelation
    # Use a chirp-like or random smooth signal
    t = np.linspace(0, 4*np.pi, N)
    h = np.sin(t) + 0.5 * np.cos(2*t) + 0.3 * np.sin(3*t + 0.7)
    # Normalize so sum is 1
    s = np.sum(h)
    if abs(s) < 1e-10:
        h = np.ones(N)
    else:
        h = h / s
    return h

def _anneal(h0, budget):
    """Simulated annealing to minimize the objective."""
    h = h0.copy()
    best_h = h.copy()
    best_val = _objective(h)
    
    start = time.time()
    T = 1.0
    T_min = 1e-6
    alpha = 0.999
    
    rng = np.random.default_rng(42)
    
    while T > T_min:
        if time.time() - start > budget:
            break
        
        # Perturbation: add small Gaussian noise
        noise_scale = T * 0.1
        h_new = h + rng.standard_normal(N) * noise_scale
        
        # Keep sum roughly constant (helps denominator)
        s_old = np.sum(h)
        s_new = np.sum(h_new)
        if abs(s_new) < 1e-10:
            continue
        # Rescale to maintain sum
        h_new = h_new * (s_old / s_new)
        
        val_new = _objective(h_new)
        val_cur = _objective(h)
        
        if val_new < val_cur or rng.random() < np.exp(-(val_new - val_cur) / T):
            h = h_new
            if val_new < best_val:
                best_val = val_new
                best_h = h.copy()
        
        T *= alpha
    
    return best_h, best_val

def _try_structural_patterns():
    """Try various structured patterns that might have low peak autocorrelation."""
    candidates = []
    
    # Pattern 1: Alternating with random phases
    for seed in range(5):
        rng = np.random.default_rng(seed)
        h = rng.standard_normal(N)
        # Make it have zero mean to reduce DC component
        h -= np.mean(h)
        if abs(np.sum(h)) < 1e-10:
            continue
        candidates.append(h)
    
    # Pattern 2: Low-pass filtered noise
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        h = rng.standard_normal(N)
        # Simple moving average filter
        kernel = np.ones(10) / 10
        h = np.convolve(h, kernel, mode='same')
        h -= np.mean(h)
        if abs(np.sum(h)) < 1e-10:
            continue
        candidates.append(h)
    
    # Pattern 3: Chirp signals
    for f1 in [1.0, 2.0, 3.0]:
        for f2 in [1.5, 2.5, 3.5]:
            t = np.linspace(0, 1, N)
            h = np.sin(2*np.pi*(f1*t + 0.5*(f2-f1)*t*t))
            h -= np.mean(h)
            if abs(np.sum(h)) < 1e-10:
                continue
            candidates.append(h)
    
    # Pattern 4: Random with sign flips in blocks
    for seed in range(3):
        rng = np.random.default_rng(200 + seed)
        h = np.ones(N)
        block_size = rng.integers(5, 20)
        for i in range(0, N, block_size):
            if rng.random() < 0.5:
                h[i:i+block_size] = -1.0
        h -= np.mean(h)
        if abs(np.sum(h)) < 1e-10:
            continue
        candidates.append(h)
    
    return candidates

def construct():
    start_time = time.time()
    
    # First, try structured patterns
    best_h = None
    best_val = np.inf
    
    for h_cand in _try_structural_patterns():
        val = _objective(h_cand)
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()
        if time.time() - start_time > _DEADLINE - 10:
            break
    
    # If we haven't found anything better than seed, try annealing
    if best_val >= 2.0 or best_h is None:
        h0 = _construct_initial()
        val0 = _objective(h0)
        if val0 < best_val:
            best_val = val0
            best_h = h0.copy()
    
    # Run annealing from the best candidate found
    if best_h is not None and time.time() - start_time < _DEADLINE - 5:
        budget = min(_DEADLINE - 5 - (time.time() - start_time), 130)
        if budget > 2:
            h_annealed, val_annealed = _anneal(best_h, budget)
            if val_annealed < best_val:
                best_val = val_annealed
                best_h = h_annealed
    
    # Also try annealing from a fresh random start if time permits
    if time.time() - start_time < _DEADLINE - 30:
        rng = np.random.default_rng(12345)
        h_rand = rng.standard_normal(N)
        h_rand -= np.mean(h_rand)
        if abs(np.sum(h_rand)) > 1e-10:
            budget = min(_DEADLINE - 30 - (time.time() - start_time), 25)
            if budget > 2:
                h2, val2 = _anneal(h_rand, budget)
                if val2 < best_val:
                    best_val = val2
                    best_h = h2
    
    # Final safety check
    if best_h is None:
        best_h = np.ones(N)
    
    # Validate
    s = np.sum(best_h)
    if abs(s) < 1e-15:
        best_h = np.ones(N)
    
    # Ensure finite
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    
    return best_h
