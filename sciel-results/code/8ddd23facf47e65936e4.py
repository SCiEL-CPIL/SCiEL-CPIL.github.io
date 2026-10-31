import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initialize with all ones
    x = np.ones(N, dtype=np.int8)
    
    def score(x):
        # Compute linear autocorrelation
        # C(k) = sum x[i] * x[i+k]
        # Using FFT for speed
        # Zero pad to 2*N
        n_fft = 1 << (2 * N - 1).bit_length()
        X = np.fft.fft(x, n=n_fft)
        C = np.fft.ifft(X * np.conj(X)).real
        # C[0:N] are the positive lags 0 to N-1
        # C[n_fft:N] are negative lags, but for real symmetric autocorr, we just take 0 to N-1
        c_pos = C[:N]
        
        S = np.sum(x)
        if S == 0:
            return float('inf')
            
        # Exclude lag 0
        # c_pos[0] is C(0) = N
        # We want max |C(k)| for k != 0
        max_off_peak = np.max(np.abs(c_pos[1:]))
        
        return max_off_peak / (S * S)

    current_score = score(x)
    best_x = x.copy()
    best_score = current_score
    
    # Local Search
    # We will perform hill climbing
    # To escape local minima, we use random restarts if stuck
    
    def local_search(x, max_iter):
        x = x.copy()
        cur_s = score(x)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Randomly select an index to flip
            i = np.random.randint(N)
            
            # Flip
            x[i] = -x[i]
            
            new_s = score(x)
            
            if new_s < cur_s:
                cur_s = new_s
            else:
                # Revert
                x[i] = -x[i]
                
        return x, cur_s

    # Run several rounds of local search with different random seeds/strategies
    # Since N=400, score calculation is fast.
    
    # Strategy 1: Pure Hill Climbing from all-ones
    x_curr = np.ones(N, dtype=np.int8)
    s_curr = score(x_curr)
    
    # Iterative improvement
    # We can also try to flip pairs to keep S constant?
    # But starting from all 1s, we can only reduce S.
    
    # Let's just do a long hill climb
    # Estimate iterations possible:
    # score() takes ~1ms?
    # 110s / 0.001s = 110,000 iterations.
    
    x_curr, s_curr = local_search(x_curr, 100000)
    
    if s_curr < best_score:
        best_score = s_curr
        best_x = x_curr.copy()
        
    # Strategy 2: Random restarts
    # Generate random sequences with high bias towards 1
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Bias: 80% ones, 20% minus ones
        # This maintains a large S
        x_rand = np.where(np.random.rand(N) < 0.8, 1, -1).astype(np.int8)
        
        x_opt, s_opt = local_search(x_rand, 5000)
        
        if s_opt < best_score:
            best_score = s_opt
            best_x = x_opt.copy()
            
    return best_x
