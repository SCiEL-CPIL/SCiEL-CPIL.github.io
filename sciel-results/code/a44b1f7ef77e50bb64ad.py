import math
import numpy as np

def construct():
    def score(U):
        U = sorted(set(U))
        if 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        # Efficient computation using NumPy for small to medium N
        U_arr = np.array(U)
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        n_sums = len(np.unique(sums))
        n_diffs = len(np.unique(diffs))
        
        ratio = n_diffs / n_sums
        if ratio <= 1:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    
    best_U = [0]
    best_score = -1
    
    # Search Geometric Progressions
    for base in range(2, 7):
        for k in range(3, 11):
            # U = {0, 1, base, base^2, ..., base^(k-1)}
            U = [0] + [base**i for i in range(1, k)]
            # Ensure 0 is in U and sorted
            U = sorted(set(U))
            
            s = score(U)
            if s > best_score:
                best_score = s
                best_U = U
                
    return best_U
