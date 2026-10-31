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
    # 1. Generate base sequence from Legendre symbols of a prime p approx N
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    # Find prime closest to N
    p = N
    while not is_prime(p):
        p += 1
    # Check if p-2 is prime and closer?
    p_minus = N
    while not is_prime(p_minus):
        p_minus -= 1
        
    if abs(p - N) > abs(p_minus - N):
        p = p_minus

    base_seq = np.array([_legendre_symbol(i + 1, p) for i in range(N)], dtype=float)
    # Ensure no zeros in sequence (Legendre symbol of 0 is 0, but i+1 starts at 1, so no 0s unless p divides i+1)
    # Since p ~ N, and i < N, i+1 < p, so no zeros.
    # However, if p < N, some i+1 might be divisible by p.
    # If p < N, we wrap around? The directive says p approx N.
    # If p < N, the sequence repeats.
    # Let's stick to p >= N to avoid repetition issues or just handle it.
    # If p < N, i+1 can be multiple of p. Legendre symbol is 0.
    # 0 is a valid sign? The task says "signed step heights". Usually +/-1.
    # If 0 appears, it breaks the "signed" assumption if we assume +/-1.
    # Let's map 0 to 1 or -1 randomly or based on neighbor.
    # Better: Pick p > N to ensure no 0s in the first N terms.
    
    p = N
    while not is_prime(p) or p <= N:
        p += 1
        
    base_seq = np.array([_legendre_symbol(i + 1, p) for i in range(N)], dtype=float)

    # 2. Simulated Annealing Local Search
    current_seq = base_seq.copy()
    current_obj = _objective(current_seq)
    best_seq = current_seq.copy()
    best_obj = current_obj
    
    max_iterations = 5000
    T = 10.0 # Initial temperature
    cooling = 0.99
    
    for iter in range(max_iterations):
        # Pick a random index to flip? Or Steepest Descent?
        # Steepest Descent is safer for "improvement" but can get stuck.
        # SA explores.
        # Let's use a mix: 90% Steepest, 10% Random?
        # Or just Steepest with random restarts if stuck?
        # Let's try Steepest Descent with a "kick" if stuck.
        
        improved = False
        best_gain = 0
        best_index = -1
        
        # Evaluate all single flips
        # To speed up, we can vectorize the objective calculation?
        # Hard to vectorize max(AC)/Sum^2.
        # We'll iterate.
        
        for i in range(N):
            current_seq[i] *= -1
            new_obj = _objective(current_seq)
            gain = current_obj - new_obj
            
            if gain > best_gain:
                best_gain = gain
                best_index = i
            else:
                current_seq[i] *= -1
                continue
            
            current_seq[i] *= -1 # Revert to check next from original state
        
        if best_index != -1 and best_gain > 1e-6:
            current_seq[best_index] *= -1
            current_obj -= best_gain
            improved = True
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_seq = current_seq.copy()
        else:
            # Stuck in local minimum. 
            # Perform a random perturbation (kick)
            # Flip 5 random signs
            for _ in range(5):
                idx = np.random.randint(N)
                current_seq[idx] *= -1
            current_obj = _objective(current_seq)
            
            # Accept kick if not too bad, or just proceed
            # To avoid infinite loops of bad kicks, we can restart from best if too many kicks
            # For now, just accept the kick and continue SA-like cooling
            T *= cooling

    return best_seq
# EVOLVE-BLOCK-END
