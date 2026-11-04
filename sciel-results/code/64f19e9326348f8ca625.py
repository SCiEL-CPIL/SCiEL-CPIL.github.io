# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 400
    
    # Initialize random binary sequence
    f = np.random.choice([-1, 1], size=N)
    
    # Compute initial autocorrelation R
    # R[k] = sum f[n] f[n+k]
    # Using FFT for initial calculation
    F = np.fft.fft(f)
    power = F * np.conj(F)
    R = np.real(np.fft.ifft(power))
    
    # PSL is max absolute value of non-zero lags
    # Note: R[0] is the peak.
    # We want to minimize max(|R[1]|, ..., |R[N-1]|)
    
    def get_psl(R):
        return np.max(np.abs(R[1:]))
        
    current_psl = get_psl(R)
    best_psl = current_psl
    best_f = f.copy()
    best_R = R.copy()
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.9999
    min_T = 1e-6
    max_iter = 500000 # Can afford more iterations with O(N) update
    
    for i in range(max_iter):
        if T < min_T:
            break
            
        idx = np.random.randint(0, N)
        old_val = f[idx]
        new_val = -old_val
        delta_val = new_val - old_val # -2 * old_val
        
        # Update R
        # R[k] = sum f[n] f[n+k]
        # Change in R[k] due to f[idx] changing by delta_val:
        # Term 1: f[idx] is the first element: f[idx] f[idx+k] -> change is delta_val * f[idx+k]
        # Term 2: f[idx] is the second element: f[idx-k] f[idx] -> change is f[idx-k] * delta_val
        # So Delta R[k] = delta_val * (f[idx+k] + f[idx-k])
        
        # We need to update R for all k.
        # k ranges 0 to N-1.
        # Indices are modulo N.
        
        # Vectorized update
        # f_shifted_forward = np.roll(f, -k)? No, we need f[idx+k] for each k.
        # Let's construct the vector of changes.
        # For a given k, term is f[(idx+k)%N] + f[(idx-k)%N]
        
        # We can compute this for all k at once.
        # k = 0..N-1
        ks = np.arange(N)
        idx_plus = (idx + ks) % N
        idx_minus = (idx - ks) % N
        
        changes = delta_val * (f[idx_plus] + f[idx_minus])
        
        R_new = R + changes
        
        new_psl = get_psl(R_new)
        delta_psl = new_psl - current_psl
        
        if delta_psl < 0 or np.random.rand() < np.exp(-delta_psl / T):
            # Accept
            f[idx] = new_val
            R = R_new
            current_psl = new_psl
            if new_psl < best_psl:
                best_psl = new_psl
                best_f = f.copy()
                best_R = R.copy()
        else:
            # Reject, R remains unchanged (we created R_new as a copy, so R is safe)
            pass
            
        T *= cooling_rate
        
    return best_f
# EVOLVE-BLOCK-END
