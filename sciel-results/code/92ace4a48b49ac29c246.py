import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 80.0
    
    N = 129
    target_lo = 128
    target_hi = 256
    
    def get_counts(x):
        return np.convolve(x, x, mode='full')
        
    def get_objective(x):
        c = np.convolve(x, x, mode='full')
        target_c = c[128:257]
        
        s = np.arange(128, 257)
        half = s // 2
        even_mask = (s % 2 == 0)
        d = np.zeros(129)
        valid_half = half < N
        d[even_mask & valid_half] = x[half[even_mask & valid_half]]
        
        v = target_c + d
        max_val = np.max(v)
        min_val = np.min(v)
        is_valid = min_val > 0
        return max_val, is_valid, v

    def get_score(x):
        max_val, is_valid, v = get_objective(x)
        if not is_valid:
            gaps = np.sum(v == 0)
            return max_val + 1000 * gaps
        return max_val

    # Initialize
    x = np.zeros(N, dtype=int)
    indices = np.random.choice(N, 60, replace=False)
    x[indices] = 1
    
    best_x = x.copy()
    best_score = get_score(best_x)
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    current_x = best_x.copy()
    C = np.convolve(current_x, current_x, mode='full')
    
    def update_C(x, C, k, adding):
        sign = 1 if adding else -1
        j = np.arange(0, N)
        s = k + j
        valid = s <= 256
        j = j[valid]
        s = s[valid]
        
        delta = 2 * x[j]
        
        mask_2k = (s == 2*k)
        if np.any(mask_2k):
             delta[mask_2k] = sign
             
        C[s] += sign * delta
        return C

    def get_score_from_C(C, x):
        s = np.arange(128, 257)
        target_c = C[128:257]
        
        half = s // 2
        even_mask = (s % 2 == 0)
        d = np.zeros(129)
        valid_half = half < N
        d[even_mask & valid_half] = x[half[even_mask & valid_half]]
        
        v = target_c + d
        max_val = np.max(v)
        min_val = np.min(v)
        
        if min_val > 0:
            return max_val
        else:
            gaps = np.sum(v == 0)
            return max_val + 1000 * gaps

    current_score = get_score_from_C(C, current_x)
    best_score = current_score
    best_x = current_x.copy()
    
    while T > T_min:
        if time.time() - start_time > deadline:
            break
            
        k = np.random.randint(0, N)
        
        adding = not current_x[k]
        
        old_C = C.copy()
        C_new = update_C(current_x, C, k, adding)
        
        old_x_k = current_x[k]
        current_x[k] = 1 if adding else 0
        
        new_score = get_score_from_C(C_new, current_x)
        
        delta = new_score - current_score
        if delta < 0 or (T > 0 and np.random.rand() < np.exp(-delta / T)):
            C = C_new
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = current_x.copy()
        else:
            C = old_C
            current_x[k] = old_x_k
            
        T *= cooling
        
    # Local Search Polish
    improved = True
    while improved:
        if time.time() - start_time > deadline:
            break
        improved = False
        for k in range(N):
            adding = not best_x[k]
            trial_x = best_x.copy()
            trial_x[k] = 1 if adding else 0
            trial_score = get_score(trial_x)
            
            if trial_score < best_score:
                best_score = trial_score
                best_x = trial_x
                improved = True
                
    result = list(best_x.nonzero()[0])
    if not result:
        return [0]
    return result

# EVOLVE-BLOCK-END
