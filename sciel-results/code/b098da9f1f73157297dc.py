import time
from math import isqrt
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # Safety margin before 90s kill

    def sieve(n):
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return np.nonzero(is_prime)[0]

    primes = sieve(10000)
    n = len(primes)
    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]

    # fronts[i] = list of (gap, length, prev_idx) sorted by gap ascending,
    # lengths strictly increasing (Pareto front)
    fronts = [None] * n
    fronts[0] = [(0, 1, -1)]
    
    best_len = 1
    best_end = 0

    # Precompute primes as numpy array for fast subtraction
    p_arr = primes  # already numpy array

    for i in range(1, n):
        if time.time() - start_time > DEADLINE:
            break
        
        pi = p_arr[i]
        # Compute all gaps from previous primes to pi in one shot
        # gaps[j] = pi - p_arr[j] for j in 0..i-1
        gaps = pi - p_arr[:i]
        
        # For each j, we need to find the best state in fronts[j] with gap_j <= gaps[j]
        # Then the candidate is (gaps[j], best_len_j + 1, j)
        
        candidates = [(0, 1, -1)]  # sequence of just primes[i]
        
        # Process each j
        for j in range(i):
            front_j = fronts[j]
            if not front_j:
                continue
            
            gap = int(gaps[j])
            
            # Binary search in front_j for rightmost entry with gap <= current gap
            # front_j is sorted by gap ascending
            lo, hi = 0, len(front_j)
            while lo < hi:
                mid = (lo + hi) // 2
                if front_j[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            
            if lo > 0:
                last_gap, length, prev_idx = front_j[lo - 1]
                candidates.append((gap, length + 1, j))
        
        # Build Pareto front: sort by gap asc, length desc, keep increasing lengths
        if len(candidates) > 1:
            candidates.sort(key=lambda x: (x[0], -x[1]))
        
        front_i = []
        max_len_so_far = 0
        for gap, length, prev_idx in candidates:
            if length > max_len_so_far:
                front_i.append((gap, length, prev_idx))
                max_len_so_far = length
        
        fronts[i] = front_i
        
        if front_i:
            if front_i[-1][1] > best_len:
                best_len = front_i[-1][1]
                best_end = i

    # Reconstruct
    if best_len <= 1:
        return [int(primes[best_end])]

    seq = []
    idx = best_end
    curr_len = best_len
    
    while curr_len > 0:
        if idx == -1:
            break
        # Find state with length curr_len in fronts[idx]
        target_state = None
        for gap, length, p_idx in fronts[idx]:
            if length == curr_len:
                target_state = (gap, length, p_idx)
                break
        
        if target_state is None:
            target_state = fronts[idx][-1]
            
        seq.append(int(primes[idx]))
        idx = target_state[2]
        curr_len -= 1
        
        if idx == -1:
            break

    seq.reverse()
    
    if not seq:
        return [int(primes[0])]
        
    return seq
