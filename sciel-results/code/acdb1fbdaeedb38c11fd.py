import numpy as np
from scipy.optimize import minimize
import time

N = 400

def construct():
    """
    Simulated Annealing approach to minimize the sum of squares of non-zero autocorrelations.
    """
    
    def compute_autocorr_sum_sq(x):
        """
        Compute sum of squares of non-zero lags of linear autocorrelation using FFT.
        x is a vector of length N with values in {-1, 1}.
        """
        x_pad = np.zeros(2 * N)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        power = X * np.conj(X)
        R_circ = np.fft.ifft(power).real
        non_central = R_circ[1:N]
        return np.sum(non_central**2)

    def compute_autocorr_sum_sq_continuous(x):
        """
        Compute sum of squares of non-zero lags of linear autocorrelation for continuous x.
        x is a vector of length N with values in [-1, 1].
        """
        x_pad = np.zeros(2 * N)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        power = X * np.conj(X)
        R_circ = np.fft.ifft(power).real
        non_central = R_circ[1:N]
        return np.sum(non_central**2)

    np.random.seed(42)
    
    # Strategy: 
    # 1. Start with random binary vector.
    # 2. Use Simulated Annealing to minimize.
    
    x = np.random.choice([-1, 1], size=N)
    current_cost = compute_autocorr_sum_sq(x)
    best_x = x.copy()
    best_cost = current_cost
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.01
    iterations = 50000 # Adjust based on time budget
    
    # Cooling schedule: Geometric
    cooling_factor = (T_end / T_start) ** (1.0 / iterations)
    
    start_time = time.time()
    
    for i in range(iterations):
        if time.time() - start_time > 100: # Leave 20s buffer
            break
            
        T = T_start * (cooling_factor ** i)
        
        # Pick a random position to flip
        idx = np.random.randint(N)
        
        # Flip
        old_val = x[idx]
        x[idx] = -old_val
        
        # Calculate new cost
        new_cost = compute_autocorr_sum_sq(x)
        delta = new_cost - current_cost
        
        # Acceptance probability
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x.copy()
        else:
            # Revert
            x[idx] = old_val
            
    return best_x

# EVOLVE-BLOCK-END
