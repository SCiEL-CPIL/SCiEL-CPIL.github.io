import numpy as np
import time

N = 400

def _autocorr_peak(h):
    # Standard FFT method for initialization
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    return np.max(np.abs(r[1:N]))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = np.random.RandomState(42)
    
    best_h = None
    best_score = float('inf')
    
    max_restarts = 50 # Can do many restarts now
    
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
            
        h = rng.choice([-1, 1], size=N).astype(np.float64)
        
        # Compute initial autocorrelation r
        n_fft = 1 << (2 * N).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        r = np.fft.irfft(power, n=n_fft)
        
        # We only care about r[1:N]
        # But we need the full r? No, just the slice for max.
        # However, for incremental updates, we only update r[1:N].
        current_score = np.max(np.abs(r[1:N]))
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # Simulated Annealing
        T = 5.0
        T_min = 0.1
        alpha = 0.999
        
        # Pre-allocate delta_r? No, we update r in place.
        
        while T > T_min and time.time() < deadline:
            i = rng.randint(N)
            
            # Save old value
            h_i_old = h[i]
            
            # Flip h
            h[i] = -h_i_old
            
            # Update r[1:N]
            # Delta r[k] = -2 * h_i_old * (h[i+k] + h[i-k])
            # We iterate k from 1 to N-1
            
            # Optimization: Vectorized update?
            # k = 1..N-1
            # idx1 = i + k
            # idx2 = i - k
            # valid1 = idx1 < N
            # valid2 = idx2 >= 0
            # term1 = np.where(valid1, h[idx1[valid1]], 0)
            # term2 = np.where(valid2, h[idx2[valid2]], 0)
            # delta = -2 * h_i_old * (term1 + term2)
            
            # This vectorization might be slower than a loop for N=400?
            # N=400 is small. A Python loop is slow. Vectorization is better.
            
            k = np.arange(1, N)
            
            idx1 = i + k
            idx2 = i - k
            
            # Create masks
            mask1 = idx1 < N
            mask2 = idx2 >= 0
            
            # Get values, 0 if out of bounds
            # Use np.take with mode? Or just indexing with where
            # h is length N.
            
            val1 = np.zeros(N-1)
            val2 = np.zeros(N-1)
            
            val1[mask1] = h[idx1[mask1]]
            val2[mask2] = h[idx2[mask2]]
            
            delta_r = -2.0 * h_i_old * (val1 + val2)
            
            r[1:N] += delta_r
            
            new_score = np.max(np.abs(r[1:N]))
            
            # Acceptance
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                delta = new_score - current_score
                if delta < 50:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                
                if rng.random() < prob:
                    current_score = new_score
                else:
                    # Reject: Undo
                    h[i] = h_i_old
                    r[1:N] -= delta_r
            
            T *= alpha

    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
