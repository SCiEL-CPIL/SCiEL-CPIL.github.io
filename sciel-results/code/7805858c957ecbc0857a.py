import numpy as np

N = 100

def construct():
    np.random.seed(42)
    
    def compute_autocorr(h):
        # Circular autocorrelation
        f = np.fft.fft(h)
        acf = np.fft.ifft(f * np.conj(f)).real
        return acf

    def get_min_overlap(acf):
        # acf[0] is N/2 = 50
        # min over k=1..N-1
        return 50 - np.min(acf[1:])

    best_h = None
    best_score = float('inf')
    
    # Random restarts
    for restart in range(50):
        h = np.zeros(N)
        ones = np.random.choice(N, N // 2, replace=False)
        h[ones] = 1.0
        
        acf = compute_autocorr(h)
        current_score = get_min_overlap(acf)
        
        improved = True
        while improved:
            improved = False
            best_swap = None
            best_new_score = current_score
            
            ones_idx = np.where(h == 1)[0]
            zeros_idx = np.where(h == 0)[0]
            
            # Try all swaps
            for i in ones_idx:
                for j in zeros_idx:
                    # Calculate new acf
                    # A'_k = A_k - h[(i+k)%N] + h[(j+k)%N]
                    # Note: h[i] is 1, h[j] is 0
                    # So we subtract the contribution of i and add contribution of j
                    
                    # Vectorized update
                    # shifts = np.arange(N)
                    # new_acf = acf - h[(i+shifts)%N] + h[(j+shifts)%N]
                    
                    # To avoid creating large arrays, we can just compute the min
                    # min_k (acf[k] - h[(i+k)%N] + h[(j+k)%N])
                    
                    # Let's just do the vector op, it's fast
                    shifts = np.arange(N)
                    new_acf = acf - h[(i + shifts) % N] + h[(j + shifts) % N]
                    new_score = 50 - np.min(new_acf[1:])
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_swap = (i, j)
            
            if best_swap is not None:
                i, j = best_swap
                h[i] = 0
                h[j] = 1
                # Update acf globally for next iteration
                shifts = np.arange(N)
                acf = acf - h[(i + shifts) % N] + h[(j + shifts) % N]
                # Wait, h[i] is now 0, h[j] is now 1.
                # The update formula was: new_acf = old_acf - old_h[(i+k)] + old_h[(j+k)]
                # Since we are updating acf in place for the next loop:
                # acf should become new_acf.
                # But we used old h in the calculation.
                # So acf = new_acf is correct.
                
                current_score = best_new_score
                improved = True
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    return best_h
