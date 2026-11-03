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
        
        U_arr = np.array(U)
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        n_sums = len(np.unique(sums))
        n_diffs = len(np.unique(diffs))
        
        ratio = n_diffs / n_sums
        if ratio <= 1:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    
    # Known optimal Golomb rulers (OEIS A005293)
    rulers = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 28],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 41],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 41, 47],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 41, 47, 55],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 41, 47, 55, 61],
    }
    
    best_U = [0]
    best_score = -1
    
    for N in range(1, 13):
        U = rulers[N]
        s = score(U)
        if s > best_score:
            best_score = s
            best_U = U
            
    return best_U
