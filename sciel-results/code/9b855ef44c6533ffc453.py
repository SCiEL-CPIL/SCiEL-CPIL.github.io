import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin before 150s kill

def _supnorm_grid(coeffs, n_grid=2048):
    """Fast sup-norm estimate using FFT on a grid."""
    # P(z) = sum c_k z^k evaluated at z = exp(2*pi*i*m/n_grid)
    # This is DFT of coeffs (padded to n_grid)
    padded = np.zeros(n_grid, dtype=complex)
    padded[:len(coeffs)] = coeffs
    vals = np.fft.fft(padded)
    return float(np.max(np.abs(vals)))

def _refine_peak(coeffs, ang, iters=60):
    """Golden-section refine max of |P(e^{i theta})| near ang."""
    n = len(coeffs)
    ks = np.arange(n)
    
    def f(theta):
        z = np.exp(1j * theta)
        # P(z) = sum c_k z^k
        return float(np.abs(np.dot(coeffs, z ** ks)))
    
    # Start with a small window around ang
    a, b = ang - 0.01, ang + 0.01
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    
    for _ in range(iters):
        if fc < fd:
            # Maximum is in [c, b]
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            # Maximum is in [a, d]
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    
    return max(f((a + b) / 2), fc, fd)

def _supnorm_precise(coeffs, n_grid=2048):
    """Sup-norm with golden-section refinement of top peaks."""
    padded = np.zeros(n_grid, dtype=complex)
    padded[:len(coeffs)] = coeffs
    vals = np.fft.fft(padded)
    mags = np.abs(vals)
    
    # Find top peaks
    idx = np.argsort(mags)[-10:][::-1]
    best = 0.0
    for i in idx:
        ang = 2 * np.pi * i / n_grid
        v = _refine_peak(coeffs, ang)
        if v > best:
            best = v
    return best

def _gr_sequence(n):
    """Generate Golay-Rudin-Shapiro sequence of length n."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    return p[:n]

def construct():
    n = 24
    
    # Generate candidate seeds
    gr = np.array(_gr_sequence(n), dtype=float)
    seeds = [
        gr,
        -gr,
        np.ones(n),
        -np.ones(n),
        np.array([1 if (k % 4 in (0, 1)) else -1 for k in range(n)], dtype=float),
        np.array([1 if (k % 3 in (0, 1)) else -1 for k in range(n)], dtype=float),
    ]
    
    # Add some random seeds
    rng = np.random.default_rng(42)
    for _ in range(20):
        seeds.append(rng.choice([-1, 1], size=n).astype(float))
    
    best = None
    best_val = float('inf')
    
    def try_vec(v):
        nonlocal best, best_val
        val = _supnorm_grid(v, n_grid=1024)
        if val < best_val * 1.01:  # Only refine if promising
            valr = _supnorm_precise(v)
            if valr < best_val:
                best_val = valr
                best = v.copy()
    
    for s in seeds:
        try_vec(s)
    
    # Local search: simulated annealing on single-bit flips
    cur = best.copy()
    cur_val = best_val
    T = 0.3
    n_iter = 0
    
    while time.time() < _DEADLINE:
        n_iter += 1
        if n_iter % 1000 == 0:
            T *= 0.995
        
        k = rng.integers(0, n)
        cand = cur.copy()
        cand[k] *= -1
        
        val = _supnorm_grid(cand, n_grid=512)
        
        if val < cur_val:
            # Accept improvement
            cur = cand
            cur_val = val
            # Occasionally refine to update global best
            if n_iter % 100 == 0:
                valr = _supnorm_precise(cand)
                if valr < best_val:
                    best_val = valr
                    best = cand.copy()
        else:
            # Accept with probability exp(-(val - cur_val)/T)
            if T > 1e-6 and rng.random() < np.exp(-(val - cur_val) / T):
                cur = cand
                cur_val = val
    
    # Final refinement
    if best is not None:
        final_val = _supnorm_precise(best)
        if final_val < best_val:
            best_val = final_val
    
    if best is None:
        best = seeds[0]
    
    return [int(x) for x in best]
