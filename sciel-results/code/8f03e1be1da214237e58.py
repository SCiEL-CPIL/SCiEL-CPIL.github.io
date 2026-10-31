import numpy as np
import time
import scipy.optimize as opt

N = 400

# EVOLVE-BLOCK-START

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _legendre_symbol(a, p):
    """
    Compute the Legendre symbol (a/p).
    Returns 1 if a is a quadratic residue mod p, -1 if non-residue, 0 if a % p == 0.
    """
    if a % p == 0:
        return 0
    res = pow(a, (p - 1) // 2, p)
    if res == 1:
        return 1
    elif res == p - 1:
        return -1
    else:
        return 0  # Should not happen for prime p

def _generate_legendre_sequence(p, N):
    """
    Generate a sequence of length N based on the Legendre symbol of a prime p.
    h[i] = Legendre(i+1, p) for i in range(N).
    If Legendre is 0, map to 1.
    """
    h = np.zeros(N)
    for i in range(N):
        sym = _legendre_symbol(i + 1, p)
        h[i] = 1.0 if sym >= 0 else -1.0
    return h

def _local_sign_flip_incremental(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves.
    Uses FFT for convolution updates.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    # Precompute FFT of h
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    max_abs_ac = np.max(np.abs(ac))
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, 1e10
    best_val = max_abs_ac / (s * s)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try flipping sign at position i
            v = h[i]
            delta = -2.0 * v
            
            h[i] = h[i] + delta
            H_new = np.fft.rfft(h, fsize)
            ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
            max_abs_new = np.max(np.abs(ac_new))
            s_new = s + delta
            if abs(s_new) < 1e-6:
                h[i] = v  # revert
                continue
            val_new = max_abs_new / (s_new * s_new)
            
            if val_new < best_val - 1e-12:
                best_val = val_new
                H = H_new
                ac = ac_new
                max_abs_ac = max_abs_new
                s = s_new
                improved = True
            else:
                h[i] = v  # revert
    
        if not improved:
            break
    
    return h, best_val

def _triple_flip_optimize(h, max_passes=3, deadline=None):
    """
    Try flipping triples of signs to escape shallow local minima.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    max_abs_ac = np.max(np.abs(ac))
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, 1e10
    best_val = max_abs_ac / (s * s)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Sample random triples to keep runtime bounded
        n_triples = min(500, n // 2)
        for _t in range(n_triples):
            if deadline is not None and time.time() > deadline:
                break
            idx = np.random.choice(n, 3, replace=False)
            deltas = np.array([-2.0 * h[i] for i in idx])
            old_vals = h[idx].copy()
            s_old = s
            s_new = s_old + np.sum(deltas)
            if abs(s_new) < 1e-6:
                continue
            h[idx] = old_vals + deltas
            H_new = np.fft.rfft(h, fsize)
            ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
            max_abs_new = np.max(np.abs(ac_new))
            val_new = max_abs_new / (s_new * s_new)
            if val_new < best_val - 1e-12:
                best_val = val_new
                H = H_new
                ac = ac_new
                max_abs_ac = max_abs_new
                s = s_new
                improved = True
            else:
                h[idx] = old_vals
        if not improved:
            break
    return h, best_val

def _smooth_surrogate_polish(h, deadline, maxiter=30):
    """
    Use a smooth surrogate (soft-max of |ac| via log-sum-exp) to escape nonsmooth plateaus.
    Minimizes logsumexp(|ac| / tau) / (sum h)^2 with small tau.
    """
    if time.time() > deadline:
        return h, _score(h)
    h0 = h.copy().astype(np.float64)
    s0 = np.sum(h0)
    if abs(s0) < 1e-6:
        return h0, _score(h0)
    h_norm = h0 / s0
    tau = 0.5  # softness
    
    def obj(x):
        if time.time() > deadline:
            return 1e10
        s = np.sum(x)
        if abs(s) < 1e-6:
            return 1e10
        size = 2 * len(x) - 1
        fsize = 1
        while fsize < size:
            fsize *= 2
        Hf = np.fft.rfft(x, fsize)
        ac = np.fft.irfft(Hf * Hf, fsize)[:size]
        # soft-max of |ac|
        a = np.abs(ac)
        m = np.max(a)
        lse = m + np.log(np.sum(np.exp((a - m) / tau))) * tau
        return lse / (s * s)
    
    try:
        res = opt.minimize(obj, h_norm, method='BFGS',
                           options={'maxiter': maxiter, 'disp': False, 'gtol': 1e-6})
        h_opt = res.x
        val = _score(h_opt)
        if val < _score(h0):
            return h_opt, val
        return h0, _score(h0)
    except Exception:
        return h0, _score(h0)

def _generate_chirp_sequence(N, T):
    """
    Generate a quadratic phase chirp sequence: h[i] = sign(cos(pi * i^2 / T)).
    Quadratic phase codes have autoconvolution sidelobes bounded by O(sqrt(N))
    due to stationary phase cancellation.
    """
    i = np.arange(N, dtype=np.float64)
    phase = np.pi * i * i / T
    h = np.sign(np.cos(phase))
    h[h == 0] = 1.0
    return h

def _bfgs_optimize(h_init, deadline, maxiter=50):
    """
    Use BFGS to optimize the real-valued sequence, minimizing the score.
    """
    if time.time() > deadline:
        return h_init, _score(h_init)
    
    # Normalize to sum 1 for stability
    s0 = np.sum(h_init)
    if abs(s0) < 1e-6:
        return h_init, _score(h_init)
    h_norm = h_init / s0
    
    def obj(x):
        if time.time() > deadline:
            return 1e10
        s = np.sum(x)
        if abs(s) < 1e-6:
            return 1e10
        return _autoconv_max_abs(x) / (s * s)
    
    try:
        res = opt.minimize(obj, h_norm, method='BFGS', 
                           options={'maxiter': maxiter, 'disp': False, 'gtol': 1e-5})
        h_opt = res.x
        val_opt = obj(h_opt)
        # Ensure sum is not zero
        if abs(np.sum(h_opt)) < 1e-6:
            return h_init, _score(h_init)
        return h_opt, val_opt
    except Exception:
        return h_init, _score(h_init)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    # Initialize best with the known baseline: all ones
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    np.random.seed(42)
    
    # Strategy:
    # 1. Use Legendre symbol sequences as initializations (algebraic structure).
    # 2. Use diverse other initializations (random, cosine, blocks).
    # 3. Apply local sign-flip optimization to each.
    # 4. Apply BFGS optimization to the best candidates.
    # 5. Keep the best.
    
    n_restarts = 300  # Try many, break on deadline
    
    # Pre-compute Legendre sequences for a few primes near N
    # Primes near 400: 397, 401, 389, 409
    primes_to_try = [397, 401, 389, 409, 383, 419, 379, 421, 373, 431]
    
    # Chirp time-scaling factors: choose T to be irrational relative to N
    # to maximize phase diversity. Use golden-ratio-based and sqrt-based values.
    chirp_Ts = [
        N / np.sqrt(2),
        N * np.sqrt(2),
        N / np.pi,
        N * np.pi,
        N * (1 + np.sqrt(5)) / 2,  # golden ratio
        N / ((1 + np.sqrt(5)) / 2),
        N * np.sqrt(3),
        N / np.sqrt(3),
        N * 0.6180339887498949,
        N * 1.618033988749895,
    ]
    
    # Store candidates for BFGS
    candidates = []
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate a diverse initialization
        if r < len(primes_to_try):
            # Legendre symbol sequence
            p = primes_to_try[r]
            h = _generate_legendre_sequence(p, N)
        elif r < len(primes_to_try) + len(chirp_Ts):
            # Quadratic phase chirp with optimized T
            T = chirp_Ts[r - len(primes_to_try)]
            h = _generate_chirp_sequence(N, T)
        elif (r - len(primes_to_try) - len(chirp_Ts)) % 5 == 0:
            # Random +/-1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif (r - len(primes_to_try) - len(chirp_Ts)) % 5 == 1:
            # Low frequency cosine: h[i] = sign(cos(2*pi*k*i/N))
            k = ((r - len(primes_to_try) - len(chirp_Ts)) // 5) % 10 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif (r - len(primes_to_try) - len(chirp_Ts)) % 5 == 2:
            # Block structure: alternating blocks of random length
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(2, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos+block_len] = sign
                pos += block_len
        elif (r - len(primes_to_try) - len(chirp_Ts)) % 5 == 3:
            # Mostly +1 with sparse -1
            h = np.ones(N)
            n_neg = np.random.randint(0, N//2)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        else:
            # Random chirp with random T
            T = np.random.uniform(0.5, 3.0) * N
            h = _generate_chirp_sequence(N, T)
        
        # Ensure the sum is not zero (which would make score inf)
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            if abs(np.sum(h)) < 1e-6:
                h[0] = 1.0
        
        # Local optimization: single flips then triple flips
        h_opt, val_opt = _local_sign_flip_incremental(h, max_passes=5, deadline=deadline)
        h_opt, val_opt = _triple_flip_optimize(h_opt, max_passes=2, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
        
        # Add to candidates for BFGS if it's a good candidate
        if val_opt < best_val * 1.1:
            candidates.append((h_opt, val_opt))
    
    # Apply BFGS and smooth surrogate to the top candidates
    # Sort by score and take the top few
    candidates.sort(key=lambda x: x[1])
    top_candidates = candidates[:10]
    
    for h_cand, val_cand in top_candidates:
        if time.time() > deadline:
            break
        
        # Try BFGS
        h_bfgs, val_bfgs = _bfgs_optimize(h_cand, deadline, maxiter=50)
        
        if val_bfgs < best_val:
            best_val = val_bfgs
            best_h = h_bfgs
        
        # Try smooth surrogate polish
        if time.time() < deadline:
            h_smooth, val_smooth = _smooth_surrogate_polish(best_h, deadline, maxiter=30)
            if val_smooth < best_val:
                best_val = val_smooth
                best_h = h_smooth
    
    # Final polish: local sign flips, triple flips, then smooth surrogate
    if time.time() < deadline:
        best_h, best_val = _local_sign_flip_incremental(best_h, max_passes=10, deadline=deadline)
    if time.time() < deadline:
        best_h, best_val = _triple_flip_optimize(best_h, max_passes=3, deadline=deadline)
    if time.time() < deadline:
        best_h, best_val = _smooth_surrogate_polish(best_h, deadline, maxiter=30)
    if time.time() < deadline:
        best_h, best_val = _local_sign_flip_incremental(best_h, max_passes=5, deadline=deadline)
    
    return best_h

# EVOLVE-BLOCK-END
