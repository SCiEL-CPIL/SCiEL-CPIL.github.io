"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _autoconv(s):
    """Compute the full autoconvolution of a 1D array."""
    # Use FFT for efficiency
    n = len(s)
    size = 2 * n - 1
    f = np.fft.rfft(s, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    return ac


def _objective(signed_seq):
    """
    Compute the maximum absolute value of the autoconvolution divided by the squared integral.
    For a sequence s, autoconv at lag k is sum_i s[i]*s[i+k].
    The 'integral' is sum(s).
    We want to minimize max(|ac|) / (sum(s))^2.
    """
    s = signed_seq
    total = np.sum(s)
    if abs(total) < 1e-9:
        return 1e9  # Avoid division by zero; large penalty
    ac = _autoconv(s)
    max_abs_ac = np.max(np.abs(ac))
    return max_abs_ac / (total * total)


def _legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p) using Euler's criterion."""
    if a % p == 0:
        return 0
    return pow(a, (p - 1) // 2, p) == 1 and 1 or -1


def construct():
    # 1. Generate a deterministic binary pattern based on the Legendre symbol of a prime near N.
    # We look for a prime p close to N.
    # Let's pick a prime p such that the sequence length is manageable.
    # The task says N=400. We can map indices 0..N-1 to Legendre symbols.
    
    # Find a prime p close to N.
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    p = N
    while not is_prime(p):
        p += 1
    # Ensure p is large enough to cover N indices if we map directly, or we can just use mod p.
    # Actually, Legendre symbol is defined for 1..p-1.
    # Let's create a base sequence of length N.
    # For i in 0..N-1, compute Legendre symbol of (i+1) mod p.
    # If p > N, then i+1 < p, so we just compute (i+1/p).
    # If p <= N, we wrap around, but Legendre symbol is periodic with p? No, it's defined mod p.
    # Let's just use a prime p > N to avoid wrap-around complications for the first N terms.
    # The next prime after 400 is 401.
    
    if p <= N:
        while p <= N:
            p += 1
            while not is_prime(p):
                p += 1

    # Generate base sequence
    base_seq = np.zeros(N, dtype=float)
    for i in range(N):
        val = _legendre_symbol(i + 1, p)
        base_seq[i] = val  # +1 or -1 (0 is not possible since i+1 < p and p is prime)

    # 2. Local sign-flip optimization
    # We will try flipping individual signs to see if it reduces the objective.
    # Since the objective is non-convex and discrete, we do a simple hill-climbing.
    # To keep it within 120s, we limit the number of iterations.
    
    current_seq = base_seq.copy()
    current_obj = _objective(current_seq)
    
    # Simple local search: try flipping each sign, keep the best improvement.
    # Repeat a few times to converge.
    max_iterations = 50
    improved = True
    iter_count = 0
    while improved and iter_count < max_iterations:
        improved = False
        best_gain = 0
        best_index = -1
        best_new_seq = None
        
        # Sample a subset of indices if N is large, but N=400 is small enough to check all.
        for i in range(N):
            new_seq = current_seq.copy()
            new_seq[i] *= -1
            new_obj = _objective(new_seq)
            gain = current_obj - new_obj
            if gain > best_gain:
                best_gain = gain
                best_index = i
                best_new_seq = new_seq
        
        if best_index != -1:
            current_seq = best_new_seq
            current_obj -= best_gain
            improved = True
        
        iter_count += 1

    return current_seq
# EVOLVE-BLOCK-END
