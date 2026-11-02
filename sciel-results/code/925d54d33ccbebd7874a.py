"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    The goal is to minimize the upper bound on C3 (third autocorrelation inequality).
    
    We model the sign sequence as the output of a deterministic chaotic map (logistic map at r=4)
    sampled at an irrational rotation number. This leverages ergodicity and sensitivity to initial
    conditions to generate a pseudo-random, low-discrepancy sequence that avoids periodic structures.
    
    The logistic map at r=4 is x_{n+1} = 4 * x_n * (1 - x_n), which is conjugate to the tent map
    and has a uniform invariant measure on [0, 1]. We use the sign of (x_n - 0.5) to determine
    the step height (+1 or -1).
    
    To improve the autoconvolution upper bound, we apply a local search refinement:
    we evaluate the C3 bound for the initial chaotic sequence, then iteratively flip
    individual bits if doing so reduces the maximum absolute value of the triple
    autocorrelation. This combines the global exploration of the chaotic map with
    local exploitation to escape local minima.
    """
    # Initialize with logistic map at r=4
    # Use an irrational initial condition to avoid periodic orbits
    x = np.sqrt(2) % 1.0  # Irrational initial condition
    
    signs = np.zeros(N)
    for i in range(N):
        x = 4.0 * x * (1.0 - x)
        # Map to sign: +1 if x > 0.5, -1 otherwise
        signs[i] = 1.0 if x > 0.5 else -1.0
    
    # Compute the C3 bound using FFT
    def compute_c3_bound(s):
        N_pad = 1 << (2 * N - 1).bit_length()
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))
    
    # Initial C3 bound
    current_bound = compute_c3_bound(signs)
    
    # Local search: flip bits to reduce C3 bound
    # We do multiple passes over the sequence, flipping bits if it improves the bound
    max_passes = 50
    for _ in range(max_passes):
        improved = False
        for i in range(N):
            # Try flipping bit i
            signs[i] *= -1.0
            new_bound = compute_c3_bound(signs)
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                # Revert the flip
                signs[i] *= -1.0
        
        if not improved:
            break
    
    return signs
# EVOLVE-BLOCK-END
