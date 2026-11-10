import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = 85.0
    
    # Phase 1: Greedy Initialization
    S = [0] * N
    for i in range(N):
        best_val = float('inf')
        best_sign = 1
        for sign in (1, -1):
            S[i] = sign
            max_d = 1
            # Optimize: Only check d up to i
            for d in range(1, i + 1):
                for r in range(d):
                    # Check if r+d <= i
                    if r + d > i:
                        continue
                    # Sequence indices: r, r+d, ... <= i
                    # Max length L such that r + (L-1)d <= i
                    # L = floor((i-r)/d) + 1
                    L = (i - r) // d + 1
                    if L < 2:
                        continue
                    # Compute max abs subarray sum of S[r::d][:L]
                    # We can do this without slicing to save time
                    max_sum = 0
                    min_sum = 0
                    cur_max = 0
                    cur_min = 0
                    idx = r
                    for _ in range(L):
                        val = S[idx]
                        cur_max = val if cur_max + val < val else cur_max + val
                        cur_min = val if cur_min + val > val else cur_min + val
                        if cur_max > max_sum:
                            max_sum = cur_max
                        if cur_min < min_sum:
                            min_sum = cur_min
                        idx += d
                    if max_sum > max_d:
                        max_d = max_sum
                    if -min_sum > max_d:
                        max_d = -min_sum
            if max_d < best_val:
                best_val = max_d
                best_sign = sign
        S[i] = best_sign
        
    # Phase 2: Precompute APs and Sums
    ap_list = []
    for d in range(1, N):
        for r in range(d):
            indices = []
            idx = r
            while idx < N:
                indices.append(idx)
                idx += d
            L = len(indices)
            if L < 2:
                continue
            for start_idx in range(L):
                for end_idx in range(start_idx + 1, L):
                    ap_list.append((indices[start_idx], d, end_idx - start_idx + 1))
    
    num_aps = len(ap_list)
    sums = [0] * num_aps
    
    for k, (start, step, length) in enumerate(ap_list):
        s = 0
        for j in range(length):
            s += S[start + j * step]
        sums[k] = s
        
    aps_containing = [[] for _ in range(N)]
    for k, (start, step, length) in enumerate(ap_list):
        for j in range(length):
            idx = start + j * step
            aps_containing[idx].append(k)
    
    current_max = max(abs(s) for s in sums)
    best_max = current_max
    
    # Phase 3: Local Search
    rng = random.Random(42)
    
    while time.time() - start_time < deadline:
        i = rng.randint(0, N - 1)
        old_sign = S[i]
        new_sign = -old_sign
        S[i] = new_sign
        
        new_max = current_max
        for k in aps_containing[i]:
            sums[k] -= 2 * old_sign
            abs_sum = abs(sums[k])
            if abs_sum > new_max:
                new_max = abs_sum
        
        if new_max < current_max or (new_max == current_max and rng.random() < 0.1):
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
        else:
            S[i] = old_sign
            for k in aps_containing[i]:
                sums[k] += 2 * old_sign

    return S
