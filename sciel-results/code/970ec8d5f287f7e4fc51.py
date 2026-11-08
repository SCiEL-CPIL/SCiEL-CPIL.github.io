import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Precompute lag indices for vectorized updates
    # Lags range from -(N-1) to N-1
    # Index in S array: lag + (N-1)
    Lags = np.arange(-(N-1), N)
    
    def compute_peak(h, S):
        # S is the autocorrelation array
        # Exclude lag 0 (index N-1)
        # Return max abs of sidelobes
        # We can copy and delete, or use masking
        # Masking is faster
        mask = np.ones(2*N - 1, dtype=bool)
        mask[N-1] = False
        return np.max(np.abs(S[mask]))

    best_h = None
    best_peak = float('inf')
    
    # 1. Initialization with multiple random restarts
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
        h = np.random.choice([-1.0, 1.0], size=N)
        
        # Compute S via FFT for speed (N=400, FFT is O(N log N))
        # Zero pad to 2*N
        fft_size = 1 << (2 * N - 1).bit_length()
        h_fft = np.fft.fft(h, n=fft_size)
        power = h_fft * np.conjugate(h_fft)
        autocorr = np.fft.ifft(power).real
        
        # Extract first N lags (0 to N-1)
        # autocorr[0:N] gives R_0 ... R_{N-1}
        # Due to symmetry, R_{-k} = R_k
        # We need full S array of length 2N-1
        S = np.zeros(2*N - 1)
        S[N-1:N] = autocorr[0:N]
        S[0:N-1] = autocorr[1:N][::-1]
        
        peak = compute_peak(h, S)
        
        if peak < best_peak:
            best_peak = peak
            best_h = h.copy()
            best_S = S.copy()
            
    # 2. Simulated Annealing
    current_h = best_h
    current_S = best_S
    current_peak = best_peak
    
    T = 1.0
    cooling_rate = 0.9999
    
    # Precompute masks? No, i changes.
    
    while time.time() < deadline:
        # Pick random index
        i = np.random.randint(N)
        hi = current_h[i]
        
        # Calculate delta_S
        # mask1: i+L valid
        # mask2: i-L valid
        
        # Vectorized calculation
        # Lags is array of shape (2N-1,)
        
        # i+L
        idx1 = i + Lags
        mask1 = (idx1 >= 0) & (idx1 < N)
        
        # i-L
        idx2 = i - Lags
        mask2 = (idx2 >= 0) & (idx2 < N)
        
        # Get values
        # Clip indices to avoid out of bounds, then mask
        idx1_clipped = np.clip(idx1, 0, N-1)
        val1 = np.where(mask1, current_h[idx1_clipped], 0.0)
        
        idx2_clipped = np.clip(idx2, 0, N-1)
        val2 = np.where(mask2, current_h[idx2_clipped], 0.0)
        
        delta_S = -2.0 * hi * (val1 + val2)
        
        new_S = current_S + delta_S
        new_peak = compute_peak(current_h, new_S)
        
        # Acceptance criterion
        if new_peak < current_peak:
            accept = True
        else:
            # Avoid division by zero or overflow if T is small
            if T > 1e-10:
                delta = new_peak - current_peak
                prob = np.exp(-delta / T)
                accept = np.random.rand() < prob
            else:
                accept = False
        
        if accept:
            current_h[i] *= -1.0
            current_S = new_S
            current_peak = new_peak
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_h = current_h.copy()
        
        T *= cooling_rate
        
    return best_h
