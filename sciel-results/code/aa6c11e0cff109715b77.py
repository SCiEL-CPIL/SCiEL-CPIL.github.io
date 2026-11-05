import time
from math import isqrt
from bisect import bisect_right

def construct():
    start_time = time.time()
    DEADLINE = 88

    def sieve(n):
        if n < 2:
            return []
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = 0
        is_prime[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = 0
        return [p for p in range(2, n + 1) if is_prime[p]]

    primes = sieve(10000)
    n = len(primes)
    if n == 0:
        return []
    if n == 1:
        return primes

    # For each prime index i, store:
    # front_gaps[i]: sorted list of gaps (last gap of sequences ending at i)
    # front_lens[i]: corresponding max lengths (strictly increasing with gaps)
    # front_prevs[i]: corresponding previous indices
    front_gaps = [[] for _ in range(n)]
    front_lens = [[] for _ in range(n)]
    front_prevs = [[] for _ in range(n)]
    
    # Base case at index 0
    front_gaps[0] = [0]
    front_lens[0] = [1]
    front_prevs[0] = [-1]
    
    best_len = 1
    best_end = 0

    for i in range(1, n):
        if time.time() - start_time > DEADLINE:
            break

        pi = primes[i]
        candidates = []
        
        for j in range(i):
            if time.time() - start_time > DEADLINE:
                break
            
            fg = front_gaps[j]
            if not fg:
                continue
            
            gap = pi - primes[j]
            
            # Find rightmost gap <= current gap
            idx = bisect_right(fg, gap)
            if idx > 0:
                best_idx = idx - 1
                prev_len = front_lens[j][best_idx]
                new_len = prev_len + 1
                candidates.append((gap, new_len, j))
        
        if not candidates:
            front_gaps[i] = [0]
            front_lens[i] = [1]
            front_prevs[i] = [-1]
            continue

        # Sort by gap asc, then length desc
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        fg = []
        fl = []
        fp = []
        max_len_so_far = 0
        
        for gap, length, prev_idx in candidates:
            if length > max_len_so_far:
                fg.append(gap)
                fl.append(length)
                fp.append(prev_idx)
                max_len_so_far = length
                
        front_gaps[i] = fg
        front_lens[i] = fl
        front_prevs[i] = fp
        
        if fl and fl[-1] > best_len:
            best_len = fl[-1]
            best_end = i

    # Reconstruction
    if best_len <= 1:
        return [primes[best_end]]
        
    seq = [primes[best_end]]
    idx = best_end
    current_len = best_len
    
    while current_len > 1:
        fg = front_gaps[idx]
        fl = front_lens[idx]
        fp = front_prevs[idx]
        found = False
        
        for k in range(len(fl)):
            if fl[k] == current_len:
                seq.append(primes[fp[k]])
                idx = fp[k]
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
        if p < 2 or p > 10000:
            is_valid = False
            break
        # Prime check
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
