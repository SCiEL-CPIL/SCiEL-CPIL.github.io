"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Initialize with a random sequence
    x = np.random.choice([-1, 1], N)
    
    def calc_autocorr(x):
        # Efficient autocorrelation using FFT
        n = len(x)
        f = np.fft.fft(x, 2*n)
        acf = np.fft.ifft(f * np.conj(f)).real[:n]
        return acf

    def max_abs_acf(acf):
        return np.max(np.abs(acf))

    current_acf = calc_autocorr(x)
    current_max = max_abs_acf(current_acf)
    
    best_x = x.copy()
    best_max = current_max
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Flip a random bit
        idx = np.random.randint(N)
        x[idx] *= -1
        
        new_acf = calc_autocorr(x)
        new_max = max_abs_acf(new_acf)
        
        delta = new_max - current_max
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_x = x.copy()
        else:
            # Revert
            x[idx] *= -1
            
        T *= cooling_rate
        
    return best_x
# EVOLVE-BLOCK-END
