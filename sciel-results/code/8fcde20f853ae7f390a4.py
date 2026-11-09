import time
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    weights = np.array([L // i for i in range(1, n + 1)], dtype=np.int64)
    
    def get_group_sums(start, end):
        size = end - start
        group_weights = weights[start:end]
        coeffs_list = np.array(np.meshgrid(*[[-1, 0, 1]] * size)).T.reshape(-1, size)
        sums = coeffs_list @ group_weights
        return sums, coeffs_list
    
    G1_sums, G1_coeffs = get_group_sums(0, 7)
    G2_sums, G2_coeffs = get_group_sums(7, 14)
    G3_sums, G3_coeffs = get_group_sums(14, 21)
    G4_sums, G4_coeffs = get_group_sums(21, 28)
    G5_sums, G5_coeffs = get_group_sums(28, 30)
    
    # Precompute S12 and S34
    S12_sums = (G1_sums[:, None] + G2_sums[None, :]).flatten()
    S34_sums = (G3_sums[:, None] + G4_sums[None, :]).flatten()
    
    S12_sorted = np.sort(S12_sums)
    S34_sorted = np.sort(S34_sums)
    
    # Precompute index mappings
    S12_sort_idx = np.argsort(S12_sums)
    S34_sort_idx = np.argsort(S34_sums)
    
    best_abs = None
    best_val = None
    best_i12 = None
    best_j34 = None
    best_s5_idx = None
    
    len_S12 = len(S12_sorted)
    len_S34 = len(S34_sorted)
    
    for s5_idx in range(len(G5_sums)):
        target = -G5_sums[s5_idx]
        
        i = 0
        j = len_S34 - 1
        
        while i < len_S12 and j >= 0:
            current_sum = S12_sorted[i] + S34_sorted[j]
            diff = current_sum - target
            
            if diff == 0:
                cands = []
                if i + 1 < len_S12:
                    d = S12_sorted[i+1] + S34_sorted[j] - target
                    if d != 0:
                        cands.append((abs(d), d, i+1, j))
                if j - 1 >= 0:
                    d = S12_sorted[i] + S34_sorted[j-1] - target
                    if d != 0:
                        cands.append((abs(d), d, i, j-1))
                if cands:
                    ma, mv, ni, nj = min(cands, key=lambda x: x[0])
                    if best_abs is None or ma < best_abs:
                        best_abs = ma
                        best_val = mv
                        best_i12 = ni
                        best_j34 = nj
                        best_s5_idx = s5_idx
                i += 1
                j -= 1
            elif diff < 0:
                if best_abs is None or abs(diff) < best_abs:
                    best_abs = abs(diff)
                    best_val = diff
                    best_i12 = i
                    best_j34 = j
                    best_s5_idx = s5_idx
                i += 1
            else:
                if best_abs is None or abs(diff) < best_abs:
                    best_abs = abs(diff)
                    best_val = diff
                    best_i12 = i
                    best_j34 = j
                    best_s5_idx = s5_idx
                j -= 1
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    if best_i12 is None:
        fallback = [0] * 30
        fallback[1] = 1
        fallback[2] = -1
        return fallback
    
    # Reconstruct coefficients
    s12_flat_idx = S12_sort_idx[best_i12]
    s34_flat_idx = S34_sort_idx[best_j34]
    
    g1_idx = s12_flat_idx // len(G2_sums)
    g2_idx = s12_flat_idx % len(G2_sums)
    g3_idx = s34_flat_idx // len(G4_sums)
    g4_idx = s34_flat_idx % len(G4_sums)
    
    coeffs = np.concatenate([
        G1_coeffs[g1_idx], 
        G2_coeffs[g2_idx], 
        G3_coeffs[g3_idx], 
        G4_coeffs[g4_idx], 
        G5_coeffs[best_s5_idx]
    ])
    
    return list(coeffs.astype(int))
