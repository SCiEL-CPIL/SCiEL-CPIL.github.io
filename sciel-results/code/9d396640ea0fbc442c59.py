import time
import random
import numpy as np
import math

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s kill

    # 1. Precompute APs
    ap_indices = []
    for d in range(1, n + 1):
        for a in range(n):
            idx = np.arange(a, n, d)
            if len(idx) > 0:
                ap_indices.append(idx)
    
    M = len(ap_indices)
    
    affected_aps = [[] for _ in range(n)]
    for k, idx in enumerate(ap_indices):
        for pos in idx:
            affected_aps[pos].append(k)
    
    affected_aps_np = [np.array(ks, dtype=np.int32) if ks else np.array([], dtype=np.int32) for ks in affected_aps]

    # 2. Initialize with Thue-Morse
    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs

    s = thue_morse_signs(n)
    
    ap_sums = np.zeros(M, dtype=np.int32)
    for k in range(M):
        ap_sums[k] = np.sum(s[ap_indices[k]])
        
    current_max = np.max(np.abs(ap_sums))
    
    best_s = s.copy()
    best_max = current_max

    rng = random.Random(42)
    
    # Simulated Annealing
    T = 5.0
    cooling_rate = 0.9995
    min_T = 0.05
    
    while time.time() < deadline - 1.0:
        i = rng.randint(0, n - 1)
        
        old_sign = s[i]
        new_sign = -old_sign
        delta = -2 * old_sign
        
        s[i] = new_sign
        ks = affected_aps_np[i]
        if len(ks) > 0:
            ap_sums[ks] += delta
        
        new_max = np.max(np.abs(ap_sums))
        
        if new_max < current_max:
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_s = s.copy()
        else:
            delta_obj = new_max - current_max
            if delta_obj == 0:
                if rng.random() < 0.5:
                    current_max = new_max
                else:
                    s[i] = old_sign
                    if len(ks) > 0:
                        ap_sums[ks] -= delta
            else:
                prob = math.exp(-delta_obj / T)
                if rng.random() < prob:
                    current_max = new_max
                else:
                    s[i] = old_sign
                    if len(ks) > 0:
                        ap_sums[ks] -= delta
        
        T *= cooling_rate
        if T < min_T:
            T = 5.0
            s = best_s.copy()
            ap_sums = np.zeros(M, dtype=np.int32)
            for k in range(M):
                ap_sums[k] = np.sum(s[ap_indices[k]])
            current_max = best_max
            
            num_flips = rng.randint(5, 15)
            for _ in range(num_flips):
                p = rng.randint(0, n - 1)
                old_s = s[p]
                s[p] = -old_s
                d_s = -2 * old_s
                ks_p = affected_aps_np[p]
                if len(ks_p) > 0:
                    ap_sums[ks_p] += d_s
            current_max = np.max(np.abs(ap_sums))

    return [int(x) for x in best_s]
