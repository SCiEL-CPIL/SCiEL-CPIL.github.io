"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Use a Legendre sequence (quadratic residues) as a base
    # This is a deterministic construction known for low autocorrelation
    p = 401  # Next prime > 400
    h = np.zeros(p, dtype=float)
    for i in range(1, p):
        # Legendre symbol
        h[i] = 1.0 if pow(i, (p - 1) // 2, p) == 1 else -1.0
    h[0] = 1.0
    
    # Truncate to N
    h = h[:N].copy()
    
    # Local search to further minimize the sum of squared autocorrelations
    def calc_cost(seq):
        n = len(seq)
        # Zero-pad to avoid circular correlation artifacts in FFT
        f = np.fft.fft(seq, 2 * n)
        acf = np.fft.ifft(f * np.conj(f)).real[:n]
        # Exclude the zero-lag term (which is constant N)
        return np.sum(acf[1:] ** 2)
        
    current_cost = calc_cost(h)
    
    # Perform a few passes of hill climbing
    for _ in range(3):
        improved = False
        for i in range(N):
            # Flip bit i
            h[i] = -h[i]
            new_cost = calc_cost(h)
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                h[i] = -h[i] # Revert
        if not improved:
            break
            
    return h
# EVOLVE-BLOCK-END
