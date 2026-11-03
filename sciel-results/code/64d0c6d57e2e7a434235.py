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
    
    # Hardcoded optimal Golomb rulers for N=1 to 10
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
        10: [0, 1, 4, 10, 18, 23, 29, 37, 41, 47]
    }
    
    def greedy_golomb(N):
        U = [0]
        diffs = set()
        for _ in range(1, N):
            x = 1
            while True:
                valid = True
                for u in U:
                    d = abs(x - u)
                    if d in diffs:
                        valid = False
                        break
                if valid:
                    # Add x
                    for u in U:
                        diffs.add(abs(x - u))
                    U.append(x)
                    break
                x += 1
        return U

    best_U = [0]
    best_score = -1
    
    # Evaluate hardcoded
    for N in range(1, 11):
        U = rulers[N]
        s = score(U)
        if s > best_score:
            best_score = s
            best_U = U
            
    # Evaluate generated for N=11 to 20
    for N in range(11, 21):
        U = greedy_golomb(N)
        s = score(U)
        if s > best_score:
            best_score = s
            best_U = U
            
    return best_U
