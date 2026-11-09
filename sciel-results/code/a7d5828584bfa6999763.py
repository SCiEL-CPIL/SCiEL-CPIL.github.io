import numpy as np
import time

N = 400

def _autocorr_sidelobe_max(x):
    """
    Calculate the maximum absolute value of the autocorrelation sidelobes.
    x is a 1D array of length N.
    """
    n = len(x)
    size = 2 * n
    fft_x = np.fft.fft(x, size=size)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real
    # Sidelobes are indices 1 to n-1 and n+1 to 2n-1
    # acf[0] is the peak
    pos_lags = acf[1:n]
    neg_lags = acf[n+1:2*n]
    max_val = max(np.max(np.abs(pos_lags)), np.max(np.abs(neg_lags)))
    return max_val

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initialize with random +/- 1 sequence
    best_x = np.random.choice([-1.0, 1.0], N)
    best_score = _autocorr_sidelobe_max(best_x)
    
    current_x = best_x.copy()
    current_score = best_score
    
    # Local search with random restarts if stuck? 
    # For now, simple hill climbing with random moves.
    
    while time.time() < deadline:
        # Try a batch of moves
        for _ in range(100):
            if time.time() > deadline:
                break
            idx = np.random.randint(N)
            new_x = current_x.copy()
            new_x[idx] *= -1.0
            
            new_score = _autocorr_sidelobe_max(new_x)
            
            if new_score < current_score:
                current_x = new_x
                current_score = new_score
                if current_score < best_score:
                    best_x = current_x.copy()
                    best_score = current_score
            else:
                # Accept worse moves with small probability to escape local minima
                # Simple: just reject for now, or use simulated annealing
                pass
                
    return best_x
