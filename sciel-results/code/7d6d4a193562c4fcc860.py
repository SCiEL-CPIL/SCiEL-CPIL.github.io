import numpy as np
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5 second safety margin
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    weights = [L // i for i in range(1, n + 1)]
    
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    def gen_sums(w):
        sums = [0]
        idxs = [0]
        for i, weight in enumerate(w):
            pow3 = 3 ** i
            new_sums = []
            new_idxs = []
            for s, idx in zip(sums, idxs):
                new_sums.append(s - weight)
                new_idxs.append(idx)
                
                new_sums.append(s)
                new_idxs.append(idx + pow3)
                
                new_sums.append(s + weight)
                new_idxs.append(idx + 2 * pow3)
            sums = new_sums
            idxs = new_idxs
        return sums, idxs

    sums1, idxs1 = gen_sums(w1)
    sums2, idxs2 = gen_sums(w2)
    
    sums1_arr = np.array(sums1, dtype=np.int64)
    idxs1_arr = np.array(idxs1, dtype=np.int64)
    sums2_arr = np.array(sums2, dtype=np.int64)
    idxs2_arr = np.array(idxs2, dtype=np.int64)
    
    order1 = np.argsort(sums1_arr)
    sorted_sums1 = sums1_arr[order1]
    sorted_idxs1 = idxs1_arr[order1]
    
    order2 = np.argsort(sums2_arr)
    sorted_sums2 = sums2_arr[order2]
    sorted_idxs2 = idxs2_arr[order2]
    
    i = 0
    j = len(sorted_sums2) - 1
    best_abs = float('inf')
    best_idx1 = -1
    best_idx2 = -1
    
    while i < len(sorted_sums1) and j >= 0:
        if time.time() > deadline:
            break
            
        s1 = sorted_sums1[i]
        s2 = sorted_sums2[j]
        total = s1 + s2
        
        if total == 0:
            # Check neighbors for non-zero
            # Neighbor 1: i+1, j
            if i + 1 < len(sorted_sums1):
                nt = sorted_sums1[i+1] + s2
                if nt != 0 and abs(nt) < best_abs:
                    best_abs = abs(nt)
                    best_idx1 = sorted_idxs1[i+1]
                    best_idx2 = sorted_idxs2[j]
            # Neighbor 2: i, j-1
            if j - 1 >= 0:
                nt = s1 + sorted_sums2[j-1]
                if nt != 0 and abs(nt) < best_abs:
                    best_abs = abs(nt)
                    best_idx1 = sorted_idxs1[i]
                    best_idx2 = sorted_idxs2[j-1]
            
            i += 1
            j -= 1
        elif total > 0:
            if abs(total) < best_abs:
                best_abs = abs(total)
                best_idx1 = sorted_idxs1[i]
                best_idx2 = sorted_idxs2[j]
            j -= 1
        else:
            if abs(total) < best_abs:
                best_abs = abs(total)
                best_idx1 = sorted_idxs1[i]
                best_idx2 = sorted_idxs2[j]
            i += 1

    if best_idx1 == -1:
        return [0]*28 + [1, -1]

    def decode(idx, k):
        coeffs = []
        rem = idx
        for j in range(k):
            digit = rem % 3
            rem //= 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
        return coeffs

    c1 = decode(best_idx1, half)
    c2 = decode(best_idx2, n - half)
    
    return c1 + c2
