import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p)."""
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
    """Generate a sequence of length N based on the Legendre symbol of a prime p."""
    h = np.zeros(N)
    for i in range(N):
        sym = _legendre_symbol(i + 1, p)
        h[i] = 1.0 if sym >= 0 else -1.0
    return h

def _local_search(h, deadline=None, max_passes=20):
    """
    Local optimization with single-flip and pair-flip moves.
    Returns (h_optimized, best_value).
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    # Compute initial score
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    max_abs_ac = np.max(np.abs(ac))
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, 1e10
    best_val = max_abs_ac / (s * s)
    
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Single flip moves
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            v = h[i]
            delta = -2.0 * v
            
            # Try flipping position i
            h[i] = v + delta
            H_new = np.fft.rfft(h, fsize)
            ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
            max_abs_new = np.max(np.abs(ac_new))
            s_new = s + delta
            
            if abs(s_new) < 1e-6:
                h[i] = v
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
                h[i] = v
        
        # Pair flip moves (try flipping pairs to escape local minima)
        if not improved:
            # Try a few random pair flips
            for _ in range(min(50, n // 2)):
                if deadline is not None and time.time() > deadline:
                    break
                
                i, j = np.random.choice(n, 2, replace=False)
                vi, vj = h[i], h[j]
                di, dj = -2.0 * vi, -2.0 * vj
                
                h[i] = vi + di
                h[j] = vj + dj
                
                H_new = np.fft.rfft(h, fsize)
                ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
                max_abs_new = np.max(np.abs(ac_new))
                s_new = s + di + dj
                
                if abs(s_new) < 1e-6:
                    h[i] = vi
                    h[j] = vj
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
                    h[i] = vi
                    h[j] = vj
        
        if not improved:
            break
    
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    # Initialize best with all ones
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    np.random.seed(42)
    
    # Pre-compute Legendre sequences for primes near N
    # Primes near 400
    primes_to_try = [397, 401, 389, 409, 383, 419, 379, 421, 373, 431, 367, 433, 359, 439, 353, 443, 349, 449, 347, 457]
    
    n_restarts = 500  # Try many, break on deadline
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate diverse initializations
        if r < len(primes_to_try):
            # Legendre symbol sequences
            p = primes_to_try[r]
            h = _generate_legendre_sequence(p, N)
        elif r % 7 == 0:
            # Random +/-1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 7 == 1:
            # Low frequency cosine
            k = (r // 7) % 15 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif r % 7 == 2:
            # Chirp-like
            x = np.arange(N)
            freq = 1.0 + (r // 7) * 0.05
            h = np.sign(np.cos(2 * np.pi * freq * x * x / (2 * N)))
            h[h == 0] = 1.0
        elif r % 7 == 3:
            # Block structure
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(2, 15)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len
        elif r % 7 == 4:
            # Mostly +1 with sparse -1
            h = np.ones(N)
            n_neg = np.random.randint(0, N // 3)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        elif r % 7 == 5:
            # Walsh-Hadamard-like pattern
            x = np.arange(N)
            k = (r // 7) % 8 + 1
            # Use bit-reversal pattern
            bits = np.zeros(N, dtype=int)
            for b in range(8):
                bits |= ((x >> b) & 1) << b
            h = np.where((bits & (1 << k)) == 0, 1.0, -1.0)
        else:
            # Quadratic phase
            x = np.arange(N)
            freq = 1.0 + (r // 7) * 0.1
            h = np.sign(np.cos(2 * np.pi * freq * x / N + np.pi * x * x / (N * N)))
            h[h == 0] = 1.0
        
        # Ensure sum is not too small
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            if abs(np.sum(h)) < 1e-6:
                h[0] = 1.0
        
        # Local optimization
        h_opt, val_opt = _local_search(h, deadline=deadline, max_passes=10)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    return best_h

# EVOLVE-BLOCK-END
