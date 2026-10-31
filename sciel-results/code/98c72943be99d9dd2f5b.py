import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def get_autocorr(s):
        S = np.fft.fft(s)
        P = S * np.conj(S)
        A = np.fft.ifft(P).real
        return A

    best_h = None
    best_score = float('inf')
    
    # Run multiple restarts
    num_restarts = 5
    
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Init
        s = np.zeros(N)
        s[:N//2] = 1
        s[N//2:] = -1
        np.random.shuffle(s)
        
        A = get_autocorr(s)
        current_max = np.max(A[1:])
        
        T = 2.0
        cooling = 0.9999 # Slow cooling
        
        # Local search within this restart
        while time.time() < deadline - 5.0 and T > 0.01:
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            if s[i] == s[j]:
                continue
                
            # Swap
            s[i], s[j] = s[j], s[i]
            
            A_new = get_autocorr(s)
            new_max = np.max(A_new[1:])
            
            delta = new_max - current_max
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_max = new_max
                A = A_new
            else:
                s[i], s[j] = s[j], s[i]
                
            T *= cooling
            
        # End of restart
        # Convert s to h
        h = (s + 1) / 2.0
        # Score is max overlap = N/2 - min C_k
        # min C_k = min (N/4 + A_k/4) = N/4 + min(A_k)/4
        # Max Overlap = N/2 - (N/4 + min(A_k)/4) = N/4 - min(A_k)/4
        # Wait, my previous derivation:
        # Score = Max Overlap = Max (N/2 - C_k) = N/2 - Min C_k
        # C_k = N/4 + A_k/4
        # Min C_k = N/4 + Min(A_k)/4
        # Score = N/2 - N/4 - Min(A_k)/4 = N/4 - Min(A_k)/4
        # We minimized Max(A_k).
        # Is Min(A_k) related to Max(A_k)?
        # Sum A_k = -A_0 = -100.
        # If Max(A_k) is small (negative), Min(A_k) must be even smaller (more negative) or similar.
        # Actually, we want to Minimize Score.
        # Score = N/4 - Min(A_k)/4.
        # To Minimize Score, we need to Maximize Min(A_k).
        # Maximize Min(A_k) is equivalent to making the distribution of A_k as flat as possible (minimizing the spread).
        # Minimizing Max(A_k) also helps flatten the distribution.
        # So minimizing Max(A_k) is a good proxy.
        
        score_val = N/4 - np.min(A[1:])/4
        if score_val < best_score:
            best_score = score_val
            best_h = h.copy()

    if best_h is None:
        return np.full(N, 0.5)
        
    return best_h
