"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_l3(h):
    """Compute sum of |R(k)|^3 for circular autocorrelation of h."""
    # Use FFT for fast autocorrelation
    H = np.fft.fft(h)
    P = H * np.conj(H)  # Power spectrum
    R = np.fft.ifft(P).real  # Circular autocorrelation
    return np.sum(np.abs(R) ** 3)

def _autocorr_l3_signed(h):
    """Compute sum of R(k)^3 (signed) for circular autocorrelation."""
    H = np.fft.fft(h)
    P = H * np.conj(H)
    R = np.fft.ifft(P).real
    return np.sum(R ** 3)

def construct():
    deadline = time.time() + 100.0  # 100 second budget, 20s safety margin
    
    # Start with random ±1 sequence
    np.random.seed(42)
    best_h = np.random.choice([-1, 1], size=N).astype(float)
    best_score = _autocorr_l3(best_h)
    
    # Simulated annealing
    h = best_h.copy()
    current_score = best_score
    
    T = 1000.0  # Initial temperature
    T_min = 0.1
    cooling = 0.9999
    
    while time.time() < deadline:
        # Pick a random position to flip
        idx = np.random.randint(N)
        old_val = h[idx]
        h[idx] = -h[idx]
        
        new_score = _autocorr_l3(h)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, T_min)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            h[idx] = old_val  # Revert
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_h
# EVOLVE-BLOCK-END
