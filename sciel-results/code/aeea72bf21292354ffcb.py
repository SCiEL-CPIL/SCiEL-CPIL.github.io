"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

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

def _compute_cost_fast(f):
    """Faster cost computation using precomputed FFT size."""
    size = 1 << (2 * N - 1).bit_length()
    f_padded = np.zeros(size)
    f_padded[:N] = f
    F = np.fft.fft(f_padded)
    R = np.fft.ifft(F * np.conj(F)).real
    return np.abs(R[0]) + 2.0 * np.sum(np.abs(R[1:N]))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Start with the seed solution (all ones)
    best_f = np.ones(N)
    best_cost = _compute_l1_autocorr(best_f)
    
    # Try a random initialization
    np.random.seed(42)
    f = np.random.choice([-1.0, 1.0], size=N)
    cost = _compute_l1_autocorr(f)
    if cost < best_cost:
        best_cost = cost
        best_f = f.copy()
    else:
        f = best_f.copy()
    
    # Simulated annealing with adaptive cooling
    T = 50.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    
    n_accept = 0
    n_reject = 0
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = np.random.randint(N)
        old_val = f[idx]
        f[idx] *= -1
        new_cost = _compute_cost_fast(f)
        
        delta = new_cost - best_cost
        
        if delta < 0:
            # Accept improvement
            best_cost = new_cost
            best_f = f.copy()
            n_accept += 1
        else:
            # Accept with probability exp(-delta/T)
            if np.random.random() < math.exp(-delta / max(T, T_min)):
                n_accept += 1
            else:
                f[idx] = old_val  # Revert
                n_reject += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best_f
# EVOLVE-BLOCK-END
