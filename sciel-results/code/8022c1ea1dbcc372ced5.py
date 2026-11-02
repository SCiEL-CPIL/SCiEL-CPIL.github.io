import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def max_abs_acorr(x):
        n_fft = 1 << (2 * N - 1).bit_length()
        X = np.fft.rfft(x, n=n_fft)
        acorr = np.fft.irfft(X * np.conj(X), n=n_fft)[:N]
        return np.max(np.abs(acorr))

    best_max = np.inf
    best_x = None
    
    # Multiple restarts
    while time.time() < deadline:
        x = np.random.choice([1, -1], size=N)
        current_max = max_abs_acorr(x)
        
        # Local search
        for _ in range(1000): # Max iterations per restart
            if time.time() > deadline:
                break
            improved = False
            # Randomly select a few indices to try, or sweep
            for i in np.random.permutation(N):
                x[i] *= -1
                new_max = max_abs_acorr(x)
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                    break
                else:
                    x[i] *= -1
                    
            if not improved:
                break
        
        if current_max < best_max:
            best_max = current_max
            best_x = x.copy()
            
    return best_x
