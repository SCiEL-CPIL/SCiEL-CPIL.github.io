import time
from math import isqrt
from bisect import bisect_right

def construct():
    start_time = time.time()
    DEADLINE = 88

    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        if n >= 0: is_prime[0] = 0
        if n >= 1: is_prime[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = 0
        return [p for p in range(2, n + 1) if is_prime[p]]

    primes = sieve(10000)
    n = len(primes)
    if n == 0: return []
    if n == 1: return primes

    # fronts[i] = (gaps_list, lens_list, prevs_list)
    fronts = [None] * n
    fronts[0] = ([0], [1], [-1])
    
    best_len = 1
    best_end = 0

    for i in range(1, n):
        if time.time() - start_time > DEADLINE:
            break

        pi = primes[i]
        candidates = [(0, 1, -1)] # (gap, length, prev_idx)
        
        for j in range(i):
            front_j = fronts[j]
            if not front_j: continue
            gaps_j, lens_j, prevs_j = front_j
            gap = pi - primes[j]
            
            # Find largest gap in gaps_j <= gap
            # bisect_right returns the index where gap would be inserted to keep order.
            # So idx - 1 is the largest element <= gap.
            idx = bisect_right(gaps_j, gap)
            if idx > 0:
                best_idx = idx - 1
                length = lens_j[best_idx] + 1
                candidates.append((gap, length, j))
        
        # Sort by gap asc, length desc
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        front_gaps = []
        front_lens = []
        front_prevs = []
        max_len_so_far = 0
        
        for gap, length, prev_idx in candidates:
            if length > max_len_so_far:
                front_gaps.append(gap)
                front_lens.append(length)
                front_prevs.append(prev_idx)
                max_len_so_far = length
                
        fronts[i] = (front_gaps, front_lens, front_prevs)
        
        if front_lens and front_lens[-1] > best_len:
            best_len = front_lens[-1]
            best_end = i

    # Reconstruction
    if best_len <= 1:
        return [primes[best_end]]
        
    target_front = fronts[best_end]
    prev_idx = -1
    for k in range(len(target_front[1])):
        if target_front[1][k] == best_len:
            prev_idx = target_front[2][k]
            break
            
    if prev_idx == -1:
        prev_idx = target_front[2][-1]
        
    seq = [primes[best_end]]
    idx = best_end
    current_len = best_len
    
    while current_len > 1:
        front_at_idx = fronts[idx]
        found = False
        for k in range(len(front_at_idx[1])):
            if front_at_idx[1][k] == current_len:
                p_idx = front_at_idx[2][k]
                seq.append(primes[p_idx])
                idx = p_idx
                current_len -= 1
                found = True
                break
        if not found:
            break
            
    seq.reverse()
    
    # Verification
    if not seq:
        return [primes[0]]
        
    is_valid = True
    for k in range(len(seq) - 1):
        if seq[k] >= seq[k + 1] or seq[k] > 10000 or seq[k] < 2:
            is_valid = False
            break
            
    for k in range(len(seq)):
        p = seq[k]
        if p < 2:
            is_valid = False
            break
        if p > 1:
            for d in range(2, isqrt(p) + 1):
                if p % d == 0:
                    is_valid = False
                    break
        if not is_valid:
            break
            
    if is_valid:
        for k in range(2, len(seq)):
            gap_curr = seq[k] - seq[k - 1]
            gap_prev = seq[k - 1] - seq[k - 2]
            if gap_curr < gap_prev:
                is_valid = False
                break
                
    if not is_valid:
        return [primes[0]]
        
    return seq

def is_prime_check(p):
    if p < 2: return False
    if p == 2: return True
    if p % 2 == 0: return False
    for d in range(3, isqrt(p) + 1, 2):
        if p % d == 0:
            return False
    return True
