"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _calc_max_side_lobe(x):
    """Calculate the maximum absolute value of the linear autocorrelation at non-zero lags."""
    # Zero-pad to length 2N-1 to compute linear autocorrelation via FFT
    x_pad = np.pad(x, (0, N - 1))
    X = np.fft.fft(x_pad)
    # Autocorrelation is IFFT of |FFT|^2
    C = np.real(np.fft.ifft(X * np.conj(X)))[:N]
    # Exclude lag 0
    return np.max(np.abs(C[1:]))

def _thue_morse_sequence(n):
    """Generate the first n elements of the Thue-Morse sequence (1, -1, -1, 1, ...)."""
    t = np.array([1])
    while len(t) < n:
        t = np.concatenate([t, -t])
    return t[:n]

def construct():
    start_time = time.time()
    # Leave 20s margin for Python overhead and termination
    deadline = start_time + 140.0
    
    # Initial guess: Thue-Morse sequence
    x = _thue_morse_sequence(N)
    current_val = _calc_max_side_lobe(x)
    best_x = x.copy()
    best_val = current_val
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.995
    
    # To make it faster, we can try to be smarter about moves, 
    # but random bit flips are simple and robust.
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = np.random.randint(N)
        old_val = x[i]
        x[i] = -old_val
        
        new_val = _calc_max_side_lobe(x)
        
        delta = new_val - current_val
        
        if delta < 0:
            # Improvement
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            # Worse or equal
            # Accept with probability exp(-delta / T)
            if np.random.random() < np.exp(-delta / T):
                current_val = new_val
            else:
                # Revert
                x[i] = old_val
        
        T *= cooling_rate
        # If T gets too small, we might get stuck. 
        # We can reheat occasionally or just let it freeze.
        if T < 0.01:
            # Reheat to escape local minima
            T = 1.0

    return best_x
# EVOLVE-BLOCK-END
