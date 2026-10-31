"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

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
    """Compute Legendre symbol (a/p) for prime p."""
    if a % p == 0:
        return 0
    return pow(a, (p - 1) // 2, p) if pow(a, (p - 1) // 2, p) == 1 else -1

def _is_prime(n):
    """Simple primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _next_prime(n):
    """Find the next prime >= n."""
    if n <= 2:
        return 2
    candidate = n if n % 2 == 1 else n + 1
    while not _is_prime(candidate):
        candidate += 2
    return candidate

def _legendre_sequence(p, length):
    """Generate a Legendre sequence of given length using prime p."""
    # The Legendre sequence is based on the Legendre symbol
    # For a prime p, the sequence has period p-1 (or p if we include the 0)
    # We'll use the first 'length' terms
    seq = np.zeros(length)
    for i in range(length):
        # Use (i+1) mod p to avoid the 0 case, or handle it
        a = (i + 1) % p
        if a == 0:
            # When i+1 is a multiple of p, Legendre symbol is 0
            # Map to +1 for our purposes
            seq[i] = 1.0
        else:
            seq[i] = float(_legendre_symbol(a, p))
    return seq

def _local_sign_flip_fast(h, max_passes=10, deadline=None):
    """
    Local optimization with incremental convolution updates for speed.
    When flipping h[i], the autoconvolution changes by -4*h[i]*h convolved with delta at i.
    Specifically, if we flip h[i] (change by -2*h[i]), the new autoconv is:
    ac_new = ac_old - 2*h[i] * (h convolved with delta_i) - 2*h[i] * (delta_i convolved with h)
    But since autoconv is symmetric, this simplifies.
    
    Actually, let's use a simpler approach: precompute the full autoconv, and when
    flipping position i, update it efficiently.
    
    For autoconvolution ac[k] = sum_j h[j]*h[j+k], flipping h[i] to -h[i] changes:
    ac_new[k] = ac[k] - 2*h[i]*h[i+k] - 2*h[i]*h[i-k] + 4*h[i]^2*(if k=0 and i==i+k==i-k)
    
    This is complex. Let's use a different approach: maintain the FFT of h, and when
    flipping h[i], update the FFT incrementally.
    
    FFT update: H_new = H_old - 2*h[i]*exp(-2*pi*j*i/N) for each frequency j
    Then ac_new = ifft(H_new^2)
    
    This is O(N) per flip for the FFT update + O(N log N) for the IFFT.
    Actually, the FFT update is O(N) and IFFT is O(N log N), so total O(N log N) per flip.
    That's the same as full recompute. Let's just do full recompute but be smart about it.
    
    Alternative: use direct convolution for small N. For N=400, direct conv is O(N^2) = 160K ops,
    which is fast. Let's try direct convolution for the local search.
    """
    h = h.copy()
    best_val = _score(h)
    
    # Precompute for incremental updates
    # Use direct convolution for speed in local search
    def _direct_autoconv_max_abs(h):
        """Direct O(N^2) autoconvolution, fast for N=400."""
        n = len(h)
        max_val = 0.0
        for k in range(-n + 1, n):
            if k >= 0:
                s = np.dot(h[k:], h[:n-k])
            else:
                s = np.dot(h[:k], h[-k:])
            if abs(s) > max_val:
                max_val = abs(s)
        return max_val
    
    # Use direct convolution for local search speed
    best_val = _direct_autoconv_max_abs(h) / (np.sum(h) ** 2)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            h[i] = -old_val
            s = np.sum(h)
            if abs(s) < 1e-6:
                h[i] = old_val
                continue
            new_val = _direct_autoconv_max_abs(h) / (s * s)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    
    # Final score with FFT for accuracy
    final_val = _score(h)
    return h, final_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    # Initialize best with the known baseline: all ones
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    # Find primes near N for Legendre sequences
    # Legendre sequences work best when p is close to N
    primes_near_N = []
    p = _next_prime(N - 50)
    while p < N + 50 and len(primes_near_N) < 10:
        primes_near_N.append(p)
        p = _next_prime(p + 1)
    
    np.random.seed(42)
    
    n_restarts = 300  # Try many, break on deadline
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate a diverse initialization
        init_type = r % 7
        if init_type == 0:
            # Legendre sequence using prime near N
            p = primes_near_N[r % len(primes_near_N)]
            h = _legendre_sequence(p, N)
        elif init_type == 1:
            # Random +/-1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif init_type == 2:
            # Low frequency cosine
            k = (r // 7) % 10 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif init_type == 3:
            # Chirp-like
            x = np.arange(N)
            freq = 1.0 + (r // 7) * 0.1
            h = np.sign(np.cos(2 * np.pi * freq * x * x / (2*N)))
            h[h == 0] = 1.0
        elif init_type == 4:
            # Block structure
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(2, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos+block_len] = sign
                pos += block_len
        elif init_type == 5:
            # Mostly +1 with sparse -1
            h = np.ones(N)
            n_neg = np.random.randint(0, N//2)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        else:
            # Legendre with random shift
            p = primes_near_N[r % len(primes_near_N)]
            full_seq = _legendre_sequence(p, 2*N)
            shift = np.random.randint(0, N)
            h = full_seq[shift:shift+N]
        
        # Ensure the sum is not zero
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            if abs(np.sum(h)) < 1e-6:
                h[0] = 1.0
        
        # Local optimization with faster direct convolution
        h_opt, val_opt = _local_sign_flip_fast(h, max_passes=5, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    return best_h

# EVOLVE-BLOCK-END
