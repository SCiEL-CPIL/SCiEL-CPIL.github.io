"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs_excl_lag0(f):
    """Compute max |autoconvolution| excluding lag 0."""
    n = len(f)
    # Full linear autoconvolution
    ac = np.correlate(f, f, mode='full')
    # ac[0] is lag -(n-1), ac[n-1] is lag 0, ac[2n-2] is lag +(n-1)
    # Exclude lag 0
    ac_excl = np.concatenate([ac[:n-1], ac[n:]])
    return np.max(np.abs(ac_excl))

def _score(f):
    """Compute the objective: max|autocorr| / (sum)^2. Lower is better."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    return _autoconv_max_abs_excl_lag0(f) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    # Try several analytical/parametric families
    
    # Family 1: Piecewise constant with 2 segments
    # f[i] = a for i < m, b for i >= m
    # Optimize over a, b, m
    # We can fix scale: set a=1, optimize b and m
    
    for m in range(1, N):
        if time.time() > deadline:
            break
        # f = [1]*m + [b]*(N-m)
        # S = m + b*(N-m)
        # R(k) for k > 0:
        #   If k < m: overlap of [1]*m and shifted [1]*m is (m-k), overlap of [b]*(N-m) with itself is (N-m-k), cross terms...
        # This is complex; let's just compute numerically for candidate b values
        
        # Try b values
        for b in [0.0, -0.5, -1.0, -1.5, 0.5, 1.0, 1.5, 2.0]:
            if time.time() > deadline:
                break
            f = np.concatenate([np.ones(m), np.full(N-m, b)])
            s = _score(f)
            if s < best_score:
                best_score = s
                best_f = f.copy()
    
    # Family 2: Random ±1 with DC offset
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        r = np.random.choice([-1.0, 1.0], size=N)
        for dc in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            f = dc + r
            s = _score(f)
            if s < best_score:
                best_score = s
                best_f = f.copy()
    
    # Family 3: Smooth sinusoidal modulation
    for freq in [1, 2, 3, 5, 10]:
        if time.time() > deadline:
            break
        t = np.arange(N)
        for amp in [0.1, 0.3, 0.5, 0.8, 1.0]:
            f = 1.0 + amp * np.sin(2 * np.pi * freq * t / N)
            s = _score(f)
            if s < best_score:
                best_score = s
                best_f = f.copy()
    
    # Family 4: Local optimization on a piecewise-constant signal with few breakpoints
    # Represent f as values at K breakpoints, linearly interpolated (but we want step function, so use nearest)
    # Actually, let's optimize the heights of a fixed partition
    
    n_segments = 10
    seg_len = N // n_segments
    # Initial: all ones
    heights = np.ones(n_segments)
    
    def obj(h):
        f = np.repeat(h, seg_len)
        if len(f) < N:
            f = np.concatenate([f, np.full(N - len(f), h[-1])])
        f = f[:N]
        s = np.sum(f)
        if abs(s) < 1e-12:
            return 1e10
        return _autoconv_max_abs_excl_lag0(f) / (s * s)
    
    for _ in range(3):
        if time.time() > deadline:
            break
        res = minimize(obj, heights, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-10})
        if res.fun < best_score:
            best_score = res.fun
            h = res.x
            f = np.repeat(h, seg_len)
            if len(f) < N:
                f = np.concatenate([f, np.full(N - len(f), h[-1])])
            best_f = f[:N].copy()
            heights = h.copy()
    
    # Family 5: Try a few more random structures
    for trial in range(20):
        if time.time() > deadline:
            break
        # Random smooth signal
        t = np.arange(N)
        coeffs = np.random.randn(5)
        f = 1.0 + sum(coeffs[i] * np.sin(2 * np.pi * (i+1) * t / N) for i in range(5))
        s = _score(f)
        if s < best_score:
            best_score = s
            best_f = f.copy()
    
    return best_f
# EVOLVE-BLOCK-END
