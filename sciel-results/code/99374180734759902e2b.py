import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Target: L = 2 * 10^7 powerful numbers
    # Number of powerfuls up to L: sum over b of floor(sqrt(L/b^3))
    # For L=2e7: b up to ~271, total ~50k powerfuls
    # Pairs: ~50k^2/2 = ~1.25e9 — too many for full outer product
    # Strategy: use chunked vectorized bincount
    
    L = 20_000_000  # 2 * 10^7
    
    # Generate powerful numbers up to L
    # Powerful: n = a^2 * b^3 where a,b >= 1
    powerfuls = set()
    max_b = int(L ** (1.0/3.0)) + 2
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
    
    powerfuls_list = np.array(sorted(powerfuls), dtype=np.int64)
    M = len(powerfuls_list)
    
    if time.time() > deadline - 60:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Count representations: for each N, count pairs (a,b) with a<=b, a+b=N, a,b powerful
    # Use chunked approach to avoid O(M^2) memory
    # We need bincount of all sums a_i + a_j for i <= j
    
    # Memory budget: each chunk of C powerfuls produces C*(C+1)/2 sums
    # For C=5000: ~12.5M sums = ~100MB in int64 — OK
    # For C=10000: ~50M sums = ~400MB — risky
    # Use C = 5000
    
    C = 5000  # chunk size
    
    # We'll accumulate counts in a dictionary-like structure using bincount
    # But N can be up to 2*L = 4e7, so bincount array of size 4e7+1 is ~320MB in int32
    # Use int32 to save memory
    
    max_N = 2 * L
    counts = np.zeros(max_N + 1, dtype=np.int32)
    
    # Process in chunks
    for start in range(0, M, C):
        end = min(start + C, M)
        chunk = powerfuls_list[start:end]
        c = len(chunk)
        
        # Generate all pair sums for this chunk: chunk[i] + chunk[j] for i <= j
        # Use broadcasting: chunk[:, None] + chunk[None, :] gives c x c matrix
        # We only need upper triangle (i <= j)
        # For memory efficiency, do it in sub-chunks if needed
        
        # c x c int64 array: 5000*5000*8 = 200MB — too much
        # Do sub-chunks of 2000
        sub_c = 2000
        for s in range(0, c, sub_c):
            e = min(s + sub_c, c)
            sub = chunk[s:e]
            sc = len(sub)
            
            # sub[:, None] + sub[None, :] : sc x sc
            sums_2d = sub[:, None].astype(np.int64) + sub[None, :].astype(np.int64)
            
            # Extract upper triangle (i <= j)
            # For a square matrix, upper triangle including diagonal
            idx = np.triu_indices(sc)
            sums_1d = sums_2d[idx]
            
            # Bincount these sums
            # sums_1d values are in [2, 2*L]
            local_counts = np.bincount(sums_1d, minlength=max_N + 1).astype(np.int32)
            counts += local_counts
            
            del sums_2d, sums_1d, local_counts
        
        if time.time() > deadline - 30:
            break
    
    if time.time() > deadline - 20:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find N with maximum count
    # N >= 2
    if max_N + 1 < 3:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # counts[N] gives number of ordered pairs (a,b) with a,b powerful, a+b=N
    # But we need unordered pairs a <= b
    # For a != b: ordered count = 2 * unordered count
    # For a == b (N even, a = N/2): ordered count includes 1 for (a,a)
    # So unordered count = (ordered_count + (1 if N even and N/2 powerful else 0)) / 2
    
    # Actually, let's just find the N with max ordered count, then extract pairs
    # The N with max ordered count will also have max unordered count (approximately)
    
    max_count = np.max(counts[2:])
    best_N = int(np.argmax(counts[2:])) + 2
    
    # Extract pairs for best_N
    powerfuls_set = set(powerfuls_list.tolist())
    pairs = []
    for a in powerfuls_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([int(a), int(b)])
    
    pairs.sort()
    
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': pairs}
