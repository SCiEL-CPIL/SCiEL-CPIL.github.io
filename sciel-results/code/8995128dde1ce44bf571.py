"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def _autoconv(s):
    """Compute the full autoconvolution of a 1D array using FFT."""
    n = len(s)
    size = 2 * n - 1
    f = np.fft.rfft(s, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    return ac


def _objective(signed_seq):
    """
    Compute the maximum absolute value of the autoconvolution divided by the squared integral.
    """
    s = signed_seq
    total = np.sum(s)
    if abs(total) < 1e-9:
        return 1e9
    ac = _autoconv(s)
    max_abs_ac = np.max(np.abs(ac))
    return max_abs_ac / (total * total)


def _legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _find_primes_near(n, count=5):
    """Find primes near n, both above and below."""
    primes = []
    # Search above
    p = n
    while len(primes) < count:
        if _is_prime(p):
            primes.append(p)
        p += 1
    # Search below
    p = n - 1
    while p >= 3 and len(primes) < count * 2:
        if _is_prime(p):
            primes.append(p)
        p -= 1
    return primes[:count]


def _local_search(seq, max_iters=500):
    """Local search with sign flips to minimize objective."""
    current_seq = seq.copy()
    current_obj = _objective(current_seq)
    
    for _ in range(max_iters):
        improved = False
        best_gain = 1e-12  # Small threshold to avoid numerical noise
        best_index = -1
        
        for i in range(N):
            current_seq[i] *= -1
            new_obj = _objective(current_seq)
            gain = current_obj - new_obj
            
            if gain > best_gain:
                best_gain = gain
                best_index = i
            
            current_seq[i] *= -1
        
        if best_index != -1:
            current_seq[best_index] *= -1
            current_obj -= best_gain
            improved = True
        
        if not improved:
            break
    
    return current_seq, current_obj


def construct():
    # Try multiple primes near N for better initial seeds
    primes = _find_primes_near(N, count=3)
    
    best_seq = None
    best_obj = float('inf')
    
    for p in primes:
        # Generate Legendre symbol sequence
        base_seq = np.array([_legendre_symbol(i + 1, p) for i in range(N)], dtype=float)
        # Replace any 0s with 1s (shouldn't happen for i < p)
        base_seq[base_seq == 0] = 1.0
        
        # Run local search from this seed
        seq, obj = _local_search(base_seq, max_iters=800)
        
        if obj < best_obj:
            best_obj = obj
            best_seq = seq
    
    return best_seq
# EVOLVE-BLOCK-END
