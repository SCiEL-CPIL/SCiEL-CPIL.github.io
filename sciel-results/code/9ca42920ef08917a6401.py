import math
import time
import numpy as np

def _score_set(U):
    """
    Fast scoring using FFT.
    """
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    L = 1
    while L <= 2 * max_u + 1:
        L *= 2
        
    a = np.zeros(L, dtype=np.float64)
    for u in U:
        a[u] = 1.0
        
    A = np.fft.fft(a)
    
    conv_sum = np.fft.ifft(A * A).real
    eps = 1e-6
    size_sum = np.sum(conv_sum > eps)
    
    A_conj = np.conj(A)
    conv_diff = np.fft.ifft(A * A_conj).real
    size_diff = np.sum(conv_diff > eps)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 1:
        score = 1.0
    else:
        score = 1 + math.log(ratio) / math.log(2 * max_u + 1)
        
    return score

def construct():
    start_time = time.time()
    deadline = 118.0 # Safety margin
    
    best_score = 0.0
    best_set = [0]
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 29],
        [0, 1, 4, 10, 18, 23, 29, 36]
    ]
    
    for U in candidates:
        if time.time() > deadline: break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
            
    # Random search + Local Search
    np.random.seed(42)
    
    # We will try to improve the best_set found so far
    # and also explore random starts.
    
    for n in range(3, 15):
        if time.time() > deadline: break
        for _ in range(20): # 20 random starts for each n
            if time.time() > deadline: break
            # Generate random set
            # Max U should be roughly n^2 for Sidon-like sets
            max_u_target = n * n + 5
            U = np.random.choice(max_u_target + 1, size=n, replace=False)
            U = list(U)
            if 0 not in U:
                U[0] = 0
                if len(set(U)) < n:
                    continue
            U.sort()
            
            current_score = _score_set(U)
            if current_score > best_score:
                best_score = current_score
                best_set = U[:]
                
            # Local search
            # Try to swap one element
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try removing one element and adding a new one
                for i in range(len(U)):
                    if time.time() > deadline: break
                    U_temp = U[:]
                    old_val = U_temp.pop(i)
                    # Try a few new values
                    # Range around old_val or random in range
                    for _ in range(5):
                        new_val = np.random.randint(0, max_u_target + 10)
                        if new_val in U_temp:
                            continue
                        U_temp.append(new_val)
                        U_temp.sort()
                        s = _score_set(U_temp)
                        if s > current_score:
                            current_score = s
                            U = U_temp
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = U[:]
                            break # Restart local search from new state? Or continue?
                    if improved:
                        break
                        
    return best_set
