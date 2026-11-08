"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_l1_autocorr(f):
    """Compute the L1 norm of the linear autocorrelation of f."""
    size = 1 << (2 * N - 1).bit_length()
    f_padded = np.zeros(size)
    f_padded[:N] = f
    F = np.fft.fft(f_padded)
    R = np.fft.ifft(F * np.conj(F)).real
    # R[0] is the peak (k=0). R[1:N] are the positive lags.
    # Due to symmetry, total L1 norm is R[0] + 2 * sum(|R[1:N]|)
    return np.abs(R[0]) + 2.0 * np.sum(np.abs(R[1:N]))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Start with the seed solution
    best_f = np.ones(N)
    best_cost = _compute_l1_autocorr(best_f)
    
    # Try a random initialization which usually has lower L1 autocorrelation
    f = np.random.choice([-1, 1], size=N)
    cost = _compute_l1_autocorr(f)
    if cost < best_cost:
        best_cost = cost
        best_f = f.copy()
    else:
        f = best_f.copy()
        
    # Local search: flip random bits to minimize L1 autocorrelation
    # This encourages cancellations in the autocorrelation sum
    for i in range(50000):
        if time.time() > deadline:
            break
            
        idx = np.random.randint(N)
        f[idx] *= -1
        new_cost = _compute_l1_autocorr(f)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_f = f.copy()
        else:
            f[idx] *= -1  # Revert
            
    return best_f
# EVOLVE-BLOCK-END
