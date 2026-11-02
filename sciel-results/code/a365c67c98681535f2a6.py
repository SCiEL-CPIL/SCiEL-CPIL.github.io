import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initialize with random +/- 1
    x = np.random.choice([-1, 1], size=N)
    
    def calculate_psl(x):
        # Zero pad to 2N for linear autocorrelation
        X = np.fft.fft(x, n=2*N)
        C = np.fft.ifft(X * X)
        c = C[:N].real
        # Exclude lag 0
        return np.max(np.abs(c[1:]))
        
    best_psl = calculate_psl(x)
    best_x = x.copy()
    
    # Simulated Annealing / Local Search
    # Temperature starts high, decays
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = np.random.randint(N)
        
        # Flip
        old_val = x[i]
        x[i] = -old_val
        
        new_psl = calculate_psl(x)
        
        delta = new_psl - best_psl
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            best_psl = new_psl
            best_x = x.copy()
        else:
            # Revert
            x[i] = old_val
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    return best_x
