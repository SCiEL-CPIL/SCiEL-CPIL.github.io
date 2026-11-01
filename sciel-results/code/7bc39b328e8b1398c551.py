"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr(h):
    """Compute aperiodic autocorrelation using FFT. Returns array of length N."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return acf


def _c3_value(h):
    """Compute C3 bound: sum of |sidelobe|^3 normalized."""
    acf = _autocorr(h)
    sidelobes = acf[1:]
    return np.sum(np.abs(sidelobes) ** 3) / (N ** 3)


def _smooth_c3(h, beta=50.0):
    """Smooth surrogate for C3 using log-sum-exp on |sidelobes|^3."""
    acf = _autocorr(h)
    sidelobes = acf[1:]
    cubes = np.abs(sidelobes) ** 3
    # Normalize to avoid overflow
    max_c = np.max(cubes)
    if max_c < 1e-15:
        return 0.0
    # Log-sum-exp: log(sum(exp(cubes/beta))) * beta / N^3
    shifted = cubes / beta - max_c / beta
    lse = max_c / beta + np.log(np.sum(np.exp(shifted)))
    return lse * beta / (N ** 3)


def _local_search_polish(h, deadline, max_rounds=50):
    """Deterministic local search: flip single bits to reduce C3."""
    best_h = h.copy()
    best_val = _c3_value(best_h)
    
    for _ in range(max_rounds):
        if time.time() >= deadline:
            break
        improved = False
        for i in range(N):
            if time.time() >= deadline:
                break
            best_h[i] = -best_h[i]
            val = _c3_value(best_h)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_h[i] = -best_h[i]  # revert
        if not improved:
            break
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    # Initialize with random ±1
    best_h = rng.choice([-1.0, 1.0], size=N)
    best_val = _c3_value(best_h)
    
    # Simulated annealing with smooth surrogate
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline - 30.0:
        if T < T_min:
            T = T_min
        
        # Generate candidate by flipping 1-5 random positions
        n_flips = rng.integers(1, 6)
        h = best_h.copy()
        indices = rng.choice(N, size=n_flips, replace=False)
        h[indices] = -h[indices]
        
        new_val_smooth = _smooth_c3(h, beta=T)
        
        # Accept if better, or with probability exp(-delta/T)
        if new_val_smooth < _smooth_c3(best_h, beta=T):
            best_h = h
            best_val = _c3_value(best_h)
        else:
            delta = new_val_smooth - _smooth_c3(best_h, beta=T)
            if delta < 0 or rng.random() < np.exp(-delta / T):
                best_h = h
                best_val = _c3_value(best_h)
        
        T *= cooling
    
    # Final deterministic local search polish
    if time.time() < deadline:
        polished_h, polished_val = _local_search_polish(best_h, deadline)
        if polished_val < best_val:
            best_h = polished_h
            best_val = polished_val
    
    return best_h
# EVOLVE-BLOCK-END
