import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s margin
    
    # Initialize with a random sparse vector
    # A low density vector will have low off-peak autocorrelation
    # Let's try density 0.1
    h = np.zeros(N)
    indices = np.random.choice(N, int(N*0.1), replace=False)
    h[indices] = 1.0
    
    best_h = h.copy()
    best_score = float('inf')
    
    # Calculate initial score
    # Assume metric: max_{k!=0} autocorr / sum(h^2)
    def calc_score(h):
        if np.all(h == 0):
            return float('inf')
        energy = np.sum(h**2)
        if energy == 0:
            return float('inf')
        # Compute autocorrelation
        # Using FFT for speed
        f = np.fft.fft(h, n=2*N)
        acf = np.fft.ifft(f * np.conj(f)).real
        # acf[0] is R(0). acf[1:N] are R(1)...R(N-1)
        # acf[N+1:] are negative lags
        # We need max of acf[1:N] and acf[N+1:2N] (which is same as 1:N)
        off_peak = np.max(np.concatenate([acf[1:N], acf[N+1:2*N]]))
        return off_peak / energy

    best_score = calc_score(h)
    
    # Coordinate Descent / Local Search
    # Try flipping bits or adjusting values
    # Since h is non-negative, we can try setting random elements to 0 or 1
    
    for i in range(10000):
        if time.time() > deadline:
            break
        
        # Randomly select an index to modify
        idx = np.random.randint(N)
        
        # Try setting to 0
        old_val = h[idx]
        h[idx] = 0.0
        s = calc_score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        else:
            h[idx] = old_val
            
        # Try setting to 1 (if it was 0)
        if old_val == 0.0:
            h[idx] = 1.0
            s = calc_score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
            else:
                h[idx] = old_val

    return list(best_h)
