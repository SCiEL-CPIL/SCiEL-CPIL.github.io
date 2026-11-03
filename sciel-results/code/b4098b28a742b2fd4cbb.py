import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Initialize with random binary sequence
    # All ones is bad for sidelobes, random is better start
    a = np.random.choice([1, -1], size=N)
    
    # Calculate initial autoconvolution
    # c[k] = sum(a[i] * a[i+k])
    # Using FFT for efficiency, though N=400 is small enough for direct
    # c = np.correlate(a, a, mode='full')
    # We need linear convolution (zero-padded) or circular?
    # Task says "autoconvolution". Usually linear for finite sequences.
    # c_full = np.convolve(a, a, mode='full')
    # c_full has length 2N-1.
    # c_full[0] = a[0]*a[0] = 1
    # c_full[N-1] = sum(a[i]^2) = N (Main peak)
    # Sidelobes are c_full[0:N-1] and c_full[N:2N-1]
    
    # We want max absolute value excluding the main peak at index N-1.
    # Due to symmetry, c_full[i] == c_full[2N-2-i].
    # We can just look at one half.
    
    def get_metric(seq):
        c = np.convolve(seq, seq, mode='full')
        # Main peak is at index N-1
        # Exclude it
        # Sidelobes: 0 to N-2 and N to 2N-2
        # Due to symmetry, max is in 0 to N-2
        # Return max absolute value
        return np.max(np.abs(c[:N-1]))
        
    # Simulated Annealing
    # Temperature starts high, decays
    T = 10.0
    cooling_rate = 0.999
    
    best_a = a.copy()
    best_val = get_metric(best_a)
    current_val = best_val
    current_a = a.copy()
    
    # Precompute indices for potential speedup? No, just random index.
    
    while time.time() < deadline:
        # Pick random index to flip
        i = np.random.randint(N)
        
        # Flip
        current_a[i] *= -1
        
        # Calculate new metric
        # For N=400, np.convolve is very fast.
        # We can optimize later if needed.
        new_val = get_metric(current_a)
        
        # Acceptance criterion
        delta = new_val - current_val
        if delta < 0:
            # Improvement
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_a = current_a.copy()
        else:
            # Accept with probability exp(-delta/T)
            # delta is positive (worse)
            if np.random.random() < np.exp(-delta / T):
                current_val = new_val
            else:
                # Revert
                current_a[i] *= -1
        
        # Cool down
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6
            
    return best_a

# EVOLVE-BLOCK-END
