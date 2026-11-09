import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    target_min = 128
    target_max = 256
    
    # Initial dense set
    k_init = 40
    A = np.zeros(n + 1, dtype=int)
    indices = np.random.choice(n + 1, k_init, replace=False)
    A[indices] = 1
    
    def evaluate(A_vec):
        conv = np.convolve(A_vec, A_vec, mode='full')
        target_counts = conv[target_min:target_max + 1]
        zeros = np.sum(target_counts == 0)
        if zeros > 0:
            max_val = np.max(target_counts)
            return 10000 + zeros * 100 + max_val
        else:
            return np.max(target_counts)
    
    current_score = evaluate(A)
    best_A = A.copy()
    best_score = current_score
    
    T = 50.0
    T_min = 1.0
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        idx = np.random.randint(0, n + 1)
        A[idx] = 1 - A[idx]
        
        new_score = evaluate(A)
        
        if new_score < current_score:
            current_score = new_score
            best_A = A.copy()
            best_score = current_score
        else:
            delta = new_score - current_score
            if delta < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if delta / T > 100:
                    accept = False
                else:
                    accept = np.random.rand() < np.exp(-delta / T)
            
            if not accept:
                A[idx] = 1 - A[idx]
        
        T *= cooling
        
    return list(np.where(best_A == 1)[0])
