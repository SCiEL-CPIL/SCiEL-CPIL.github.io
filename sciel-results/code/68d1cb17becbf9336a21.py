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


def construct():
    # 1. Generate base sequence from Legendre symbols of a prime p > N
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    p = N
    while not is_prime(p) or p <= N:
        p += 1

    base_seq = np.array([_legendre_symbol(i + 1, p) for i in range(N)], dtype=float)

    # 2. Aggressive Local Search
    # We want to converge to a state with high sum (denominator) and low max_ac (numerator).
    # The all-ones sequence is the global minimum for this objective if unconstrained.
    # We use a hill-climbing approach with a high iteration limit.
    
    current_seq = base_seq.copy()
    current_obj = _objective(current_seq)
    
    max_iterations = 2000
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        best_gain = 0
        best_index = -1
        
        # To speed up, we can pre-compute the autocorrelation of the current sequence
        # and update it incrementally? 
        # For N=400, full FFT is ~10us. 400 flips * 2000 iters = 800,000 FFTs.
        # 800,000 * 10us = 8 seconds. This is within the 120s limit.
        
        for i in range(N):
            # Flip sign at i
            current_seq[i] *= -1
            new_obj = _objective(current_seq)
            gain = current_obj - new_obj
            
            if gain > best_gain:
                best_gain = gain
                best_index = i
            else:
                # Revert if not the best candidate yet, or if we move on
                current_seq[i] *= -1
                continue
            
            # If this is a new best, we keep it flipped for now, 
            # but we must revert before checking the next index to ensure 
            # we are evaluating single flips from the *current* state, 
            # or we can just track the best and apply it at the end.
            # The previous code copied the array. That's expensive.
            # Let's revert and apply the best at the end of the loop.
            current_seq[i] *= -1
        
        if best_index != -1:
            current_seq[best_index] *= -1
            current_obj -= best_gain
            improved = True
        
        iter_count += 1

    return current_seq
# EVOLVE-BLOCK-END
