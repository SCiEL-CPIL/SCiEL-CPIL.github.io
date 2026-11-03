import numpy as np
import time

N = 400

def compute_acf(s):
    """Compute circular autocorrelation using FFT for initial setup."""
    fft_s = np.fft.fft(s)
    power = np.abs(fft_s) ** 2
    acf = np.real(np.fft.ifft(power))
    return np.round(acf).astype(int)

def construct():
    deadline = time.time() + 120
    
    # Initialize
    s = np.random.choice([-1, 1], N)
    R = compute_acf(s)
    
    # Calculate initial score (Normalized PSL)
    # Exclude lag 0
    max_side = np.max(np.abs(np.delete(R, 0)))
    current_score = max_side / N
    best_score = current_score
    best_s = s.copy()
    
    no_improve_count = 0
    
    while time.time() < deadline - 5:
        improved = False
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline - 5:
                break
            
            # Calculate delta R
            # Delta R[k] = -2 * s[i] * (s[(i+k)%N] + s[(i-k)%N])
            # We can vectorize this.
            # k ranges from 0 to N-1
            # s[(i+k)%N] is just a rotation of s
            # s[(i-k)%N] is a rotation of s reversed? No, just negative index.
            
            # Let's use numpy slicing/rolling
            # s_plus = roll(s, -i) -> s_plus[k] = s[(i+k)%N]
            # s_minus = roll(s, i) -> s_minus[k] = s[(i-k)%N]
            
            s_plus = np.roll(s, -i)
            s_minus = np.roll(s, i)
            
            delta_R = -2 * s[i] * (s_plus + s_minus)
            
            R_new = R + delta_R
            
            # Calculate new score
            # R_new[0] is N (unchanged? No, R[0] = sum s_i^2 = N. 
            # Delta R[0] = -2 s_i (s_i + s_i) = -4 s_i^2 = -4.
            # Wait.
            # R[0] = sum s_j s_j = N.
            # If we flip s_i, s_i becomes -s_i.
            # New R[0] = sum s_j^2 = N. It doesn't change.
            # My formula for delta_R:
            # R[k] = sum s_j s_{j+k}
            # Term j=i: s_i s_{i+k}. New: -s_i s_{i+k}. Diff: -2 s_i s_{i+k}.
            # Term j=i-k: s_{i-k} s_i. New: s_{i-k} (-s_i). Diff: -2 s_{i-k} s_i.
            # For k=0:
            # Term j=i: s_i s_i = 1. New: 1. Diff 0?
            # Formula: -2 s_i (s_i + s_i) = -4 s_i^2 = -4.
            # But R[0] should not change.
            # Ah, the sum for R[0] is sum s_j^2.
            # The "Term j=i" and "Term j=i-k" are the SAME term when k=0.
            # So for k=0, the change is 0.
            # For k != 0, the two terms are distinct.
            
            # So delta_R[0] should be 0.
            # My formula gives -4.
            # So I must set delta_R[0] = 0.
            
            delta_R[0] = 0
            
            R_new = R + delta_R
            
            # Score is max |R_new[k]| for k != 0
            # We can check if the max changed.
            # Simple: compute max.
            max_side_new = np.max(np.abs(np.delete(R_new, 0)))
            new_score = max_side_new / N
            
            if new_score < current_score:
                current_score = new_score
                R = R_new
                s[i] = -s[i]
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_s = s.copy()
            # else: revert (do nothing to s, R remains old)
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 10:
                # Perturb
                num_flips = max(1, N // 10)
                flip_indices = np.random.choice(N, num_flips, replace=False)
                s[flip_indices] *= -1
                R = compute_acf(s)
                max_side = np.max(np.abs(np.delete(R, 0)))
                current_score = max_side / N
                if current_score < best_score:
                    best_score = current_score
                    best_s = s.copy()
                no_improve_count = 0
        else:
            no_improve_count = 0

    return best_s
