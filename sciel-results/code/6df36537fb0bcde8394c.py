import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize with random +/- 1 sequence
    np.random.seed(42)
    h = np.random.choice([1, -1], N)
    
    best_h = h.copy()
    best_score = float('inf')
    
    def compute_autocorr(seq):
        """Compute linear autocorrelation using FFT."""
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(seq, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        # Extract linear autocorrelation lags 0 to N-1
        # acf[0] is lag 0
        # acf[1] is lag 1
        # ...
        # acf[N-1] is lag N-1
        # The negative lags are at the end, but due to symmetry for real signals,
        # max(abs(acf[0:N])) is sufficient for the max magnitude.
        return acf[:N]

    def get_max_sidelohe(R):
        """Get max absolute value of non-zero lag autocorrelation."""
        # R[0] is lag 0 (energy), R[1:] are lags 1..N-1
        return np.max(np.abs(R[1:]))

    current_R = compute_autocorr(h)
    current_max = get_max_sidelohe(current_R)
    best_score = current_max
    
    no_improve_count = 0
    max_no_improve = 50
    
    # Pre-compute indices for shuffling
    indices = np.arange(N)
    
    while time.time() < deadline:
        improved = False
        # Shuffle indices to avoid bias
        np.random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Calculate delta R for flipping h[i]
            delta = np.zeros(N-1)
            
            # Contribution from h[i-k] for k=1..i
            if i > 0:
                # h[i-1], h[i-2], ..., h[0]
                delta[:i] += -2.0 * h[i] * h[i-1 : -1 : -1]
            
            # Contribution from h[i+k] for k=1..N-1-i
            if i < N-1:
                # h[i+1], h[i+2], ..., h[N-1]
                delta[:N-1-i] += -2.0 * h[i] * h[i+1 : N]
            
            # Update R
            R_new = current_R + delta
            
            # Calculate new max sidelobe
            new_max = get_max_sidelohe(R_new)
            
            if new_max < current_max - 1e-9:
                # Accept flip
                h[i] *= -1
                current_R = R_new
                current_max = new_max
                improved = True
                
                if current_max < best_score - 1e-9:
                    best_score = current_max
                    best_h = h.copy()
                    no_improve_count = 0
                else:
                    # Improved current state, but not global best yet
                    # Reset no_improve counter? Or keep it?
                    # Usually reset if we make progress in local search.
                    no_improve_count = 0
            else:
                # Revert (we didn't modify h or current_R yet)
                pass
                
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random restart
                h = np.random.choice([1, -1], N)
                current_R = compute_autocorr(h)
                current_max = get_max_sidelohe(current_R)
                if current_max < best_score:
                    best_score = current_max
                    best_h = h.copy()
                    no_improve_count = 0
                else:
                    no_improve_count = 0 # Reset counter for new restart

    return best_h
