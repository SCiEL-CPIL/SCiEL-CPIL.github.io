import numpy as np
import time

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
        return 0

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

def _local_search(h, max_passes=10, deadline=None, tolerance=1e-12):
    """
    Local optimization: try flipping each element's sign, keep if improves.
    Uses full FFT recomputation for each flip candidate (simple and robust).
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
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            v = h[i]
            delta = -2.0 * v
            
            h[i] = v + delta
            s_new = s + delta
            if abs(s_new) < 1e-6:
                h[i] = v
                continue
            
            H_new = np.fft.rfft(h, fsize)
            ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
            max_abs_new = np.max(np.abs(ac_new))
            val_new = max_abs_new / (s_new * s_new)
            
            if val_new < best_val - tolerance:
                best_val = val_new
                H = H_new
                ac = ac_new
                max_abs_ac = max_abs_new
                s = s_new
                improved = True
            else:
                h[i] = v
        
        if not improved:
            break
    
    return h, best_val

def _polish_search(h, deadline=None, tolerance=1e-14):
    """
    High-precision polish: try single flips AND double flips (pairs) with tighter tolerance.
    This can escape shallow local minima that single-flip search gets stuck in.
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
    
    # First, do a few passes of single-flip polish
    for _ in range(3):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            v = h[i]
            delta = -2.0 * v
            h[i] = v + delta
            s_new = s + delta
            if abs(s_new) < 1e-6:
                h[i] = v
                continue
            H_new = np.fft.rfft(h, fsize)
            ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
            max_abs_new = np.max(np.abs(ac_new))
            val_new = max_abs_new / (s_new * s_new)
            if val_new < best_val - tolerance:
                best_val = val_new
                H = H_new
                ac = ac_new
                max_abs_ac = max_abs_new
                s = s_new
                improved = True
            else:
                h[i] = v
        if not improved:
            break
    
    # Then try double flips (pairs) to escape local minima
    for _ in range(2):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, n):
                if deadline is not None and time.time() > deadline:
                    break
                v_i, v_j = h[i], h[j]
                h[i] = -v_i
                h[j] = -v_j
                s_new = s - 2.0 * v_i - 2.0 * v_j
                if abs(s_new) < 1e-6:
                    h[i] = v_i
                    h[j] = v_j
                    continue
                H_new = np.fft.rfft(h, fsize)
                ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
                max_abs_new = np.max(np.abs(ac_new))
                val_new = max_abs_new / (s_new * s_new)
                if val_new < best_val - tolerance:
                    best_val = val_new
                    H = H_new
                    ac = ac_new
                    max_abs_ac = max_abs_new
                    s = s_new
                    improved = True
                else:
                    h[i] = v_i
                    h[j] = v_j
            if improved:
                break  # restart pair search after improvement
        if not improved:
            break
    
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Initialize best with the known baseline: all ones
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    np.random.seed(42)
    
    # Strategy:
    # 1. Use Legendre symbol sequences as initializations (algebraic structure).
    # 2. Use chirp (quadratic phase) sequences - these have known good autocorrelation properties.
    # 3. Use diverse other initializations (random, cosine, blocks).
    # 4. Apply local sign-flip optimization to each.
    # 5. Final high-precision polish pass on the best candidate.
    
    n_restarts = 300
    
    # Pre-compute Legendre sequences for a few primes near N
    primes_to_try = [397, 401, 389, 409, 383, 419, 379, 421, 373, 431, 367, 439, 359, 443, 353, 449]
    
    # Chirp frequencies to try (quadratic phase: h[i] = sign(cos(pi * a * i^2 / N)))
    chirp_freqs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.5, 1.5, 2.5, 3.5]
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate a diverse initialization
        if r < len(primes_to_try):
            # Legendre symbol sequence
            p = primes_to_try[r]
            h = _generate_legendre_sequence(p, N)
        elif r < len(primes_to_try) + len(chirp_freqs):
            # Chirp (quadratic phase) sequence
            a = chirp_freqs[r - len(primes_to_try)]
            x = np.arange(N)
            h = np.sign(np.cos(np.pi * a * x * x / N))
            h[h == 0] = 1.0
        elif r % 5 == 0:
            # Random +/-1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 5 == 1:
            # Low frequency cosine: h[i] = sign(cos(2*pi*k*i/N))
            k = (r // 5) % 10 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif r % 5 == 2:
            # Chirp-like with random frequency
            x = np.arange(N)
            freq = 0.5 + np.random.random() * 8.0
            h = np.sign(np.cos(np.pi * freq * x * x / N))
            h[h == 0] = 1.0
        elif r % 5 == 3:
            # Block structure: alternating blocks of random length
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(2, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos+block_len] = sign
                pos += block_len
        else:
            # Mostly +1 with sparse -1
            h = np.ones(N)
            n_neg = np.random.randint(0, N // 2)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        
        # Ensure the sum is not zero
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            if abs(np.sum(h)) < 1e-6:
                h[0] = 1.0
        
        # Local optimization
        h_opt, val_opt = _local_search(h, max_passes=5, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Final high-precision polish on the best candidate
    if time.time() < deadline - 10:
        best_h, best_val = _polish_search(best_h, deadline=deadline)
    
    return best_h

# EVOLVE-BLOCK-END
