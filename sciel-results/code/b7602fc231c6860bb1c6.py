import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    # Precompute powerful numbers up to a moderate limit
    L = 2 * 10**8
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
    
    powerfuls_list = sorted(powerfuls)
    M = len(powerfuls_list)
    powerfuls_set = powerfuls
    
    if time.time() > deadline - 60:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Convert to numpy for vectorized pair counting
    P = np.array(powerfuls_list, dtype=np.int64)
    max_sum = 2 * L
    
    # Use chunked bincount to count sums
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    chunk_size = 2000
    
    for i_start in range(0, M, chunk_size):
        i_end = min(i_start + chunk_size, M)
        chunk = P[i_start:i_end, None] + P[None, :]
        flat = chunk.ravel()
        chunk_counts = np.bincount(flat, minlength=max_sum + 1)
        counts += chunk_counts[:max_sum + 1]
        
        if time.time() > deadline - 40:
            break
    
    # Find top candidates from the precomputed range
    k = min(100, len(counts))
    top_indices = np.argpartition(counts, -k)[-k:]
    
    best_N = 2
    best_pairs = [[1, 1]]
    
    # Sort top indices by count descending for better early results
    top_indices = top_indices[np.argsort(counts[top_indices])[::-1]]
    
    for N in top_indices[:50]:
        N = int(N)
        if N < 2:
            continue
        pairs = []
        for a in powerfuls_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerfuls_set:
                pairs.append([a, b])
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N
            
        if time.time() > deadline - 30:
            break
    
    # Extended search: for top candidates, check powerful numbers > L
    extended_best_N = best_N
    extended_best_pairs = best_pairs
    
    for N in top_indices[:20]:
        N = int(N)
        if N < 2 or N > 2 * L:
            continue
        
        all_pairs = []
        
        # Pairs where both a, b <= L
        for a in powerfuls_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerfuls_set:
                all_pairs.append([a, b])
        
        # Pairs where a > L
        max_t = int(N ** (1.0/3.0)) + 1
        found_large = set()
        
        for t in range(1, max_t + 1):
            t3 = t * t * t
            min_s = isqrt(L // t3) + 1 if t3 <= L else 1
            max_s = isqrt(N // t3)
            
            for s in range(min_s, max_s + 1):
                a = s * s * t3
                if a > N:
                    break
                b = N - a
                if b < 1:
                    continue
                if b in powerfuls_set:
                    if a <= b:
                        pair = [a, b]
                    else:
                        pair = [b, a]
                    if pair not in all_pairs:
                        all_pairs.append(pair)
                        found_large.add(a)
                        
            if time.time() > deadline - 20:
                break
        
        if len(all_pairs) > len(extended_best_pairs):
            extended_best_pairs = all_pairs
            extended_best_N = N
            
        if time.time() > deadline - 20:
            break
    
    if not extended_best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    extended_best_pairs.sort()
    
    return {'N': extended_best_N, 'pairs': extended_best_pairs}
