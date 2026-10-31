"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


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
    a = a % p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def construct():
    # 1. Generate base sequence from Legendre symbols of a prime p >= N
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    p = N
    while not is_prime(p):
        p += 1

    # Use the first N Legendre symbols for 1..N
    base_seq = np.array([_legendre_symbol(i + 1, p) for i in range(N)], dtype=float)
    
    # Ensure the sum is not zero. Legendre sequences are balanced, so sum is small.
    # The objective divides by total^2. If total is small, objective is large.
    # We want to maximize the sum (denominator) and minimize max_ac (numerator).
    # Flipping signs to make all +1 would maximize sum but increase max_ac.
    # The Legendre sequence has low max_ac.
    
    current_seq = base_seq.copy()
    current_obj = _objective(current_seq)
    
    # 2. Local Search: Greedy Best-First
    # We try to find the single flip that improves the objective the most.
    # Repeat until no improvement is found.
    
    max_iterations = 500 # Safe limit
    improved = True
    iter_count = 0
    
    start_time = time.time()
    
    while improved and iter_count < max_iterations:
        if time.time() - start_time > 100: # Safety timeout
            break
            
        improved = False
        best_gain = 1e-9 # Small threshold to avoid infinite loops with tiny gains
        best_index = -1
        
        # Evaluate all single flips
        for i in range(N):
            # Flip
            current_seq[i] *= -1
            new_obj = _objective(current_seq)
            gain = current_obj - new_obj
            
            if gain > best_gain:
                best_gain = gain
                best_index = i
            
            # Unflip
            current_seq[i] *= -1
        
        if best_index != -1:
            # Apply the best flip
            current_seq[best_index] *= -1
            # Recalculate objective correctly
            current_obj = _objective(current_seq)
            improved = True
        
        iter_count += 1

    return current_seq
# EVOLVE-BLOCK-END
