import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    while time.time() < deadline:
        # Initialize
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search with O(N) updates
        # Flip s[i] -> -s[i]
        # Delta A_k = -2 * s[i] * (s[i+k] + s[i-k])
        # Note: s[i-k] is 0 if i-k < 0. s[i+k] is 0 if i+k >= N.
        
        improved = True
        passes = 0
        max_passes = 100
        
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            # Random order to avoid bias
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Calculate delta
                # We need to update A in place or create a copy?
                # Creating a copy is O(N). Updating in place and reverting is O(N).
                # Let's try in-place.
                
                old_si = s[i]
                # s[i] becomes -old_si
                
                # Delta for each k
                # A_new[k] = A[k] - 2 * old_si * (s[i+k] + s[i-k])
                
                # Vectorized calculation of delta
                # s[i+k] for k=0..N-1-i
                # s[i-k] for k=0..i
                
                delta = np.zeros(N)
                
                # Part 1: k where i+k < N
                # k ranges from 0 to N-1-i
                # s[i+k] is s[i : N]
                # delta[k] += -2 * old_si * s[i+k]
                # This adds to A[k]
                
                # Part 2: k where i-k >= 0
                # k ranges from 0 to i
                # s[i-k] is s[0 : i+1] reversed?
                # s[i-k] for k=0 is s[i]. For k=i is s[0].
                # So s[i-k] corresponds to s[i::-1]
                
                # Let's construct the delta array directly.
                # For a specific k, the change is -2 * s[i] * (s[i+k] + s[i-k])
                
                # This is hard to vectorize simply across all k at once without broadcasting issues.
                # Given N=400, a Python loop over k for each i is 400*400 = 160k ops per pass.
                # 160k ops in Python is ~0.1s.
                # 100 passes = 10s.
                # This is acceptable.
                
                # However, we can just recompute A using FFT if the Python loop is too slow.
                # Let's stick to the FFT re-evaluation for simplicity and reliability, 
                # but limit the number of iterations.
                
                # Actually, let's just use the random greedy FFT approach from before.
                # It is simpler and less prone to implementation errors in the delta logic.
                pass
        
        # If we don't use the O(N) update, we fall back to the random greedy FFT loop.
        
        # Let's implement the Random Greedy FFT loop here properly.
        
        restart_start = time.time()
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() - restart_start < 1.0 and no_improve < max_no_improve:
            i = np.random.randint(N)
            old_val = s[i]
            s[i] = -old_val
            
            A_new = _calc_acf(s)
            val_new = _max_abs_acf_from_array(A_new)
            
            if val_new < current_val:
                current_val = val_new
                A = A_new
                no_improve = 0
            else:
                s[i] = old_val
                no_improve += 1
                
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()

    return best_seq
