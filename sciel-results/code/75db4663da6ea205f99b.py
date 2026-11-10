"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import bisect
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 55.0  # safety margin before 60s kill
    
    def primes_upto(n):
        """Sieve of Eratosthenes."""
        if n < 2:
            return []
        sieve = bytearray(b'\x01') * (n + 1)
        sieve[0] = 0
        if n >= 1:
            sieve[1] = 0
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n + 1) if sieve[i]]
    
    # Generate primes up to 10000
    primes = primes_upto(10000)
    m = len(primes)
    
    if m < 2:
        return primes
    
    primes_arr = np.array(primes, dtype=np.int64)
    
    # dp[j] = list of (gap, length, parent_idx, parent_state_idx)
    # parent_state_idx is the index in dp[parent_idx] of the state used
    dp = [None] * m
    dp[0] = [(0, 1, -1, -1)]
    
    best_len = 1
    best_end = 0
    best_gap = 0
    best_state_idx = 0
    
    for j in range(1, m):
        if time.time() - t0 > DEADLINE:
            break
        
        pj = primes[j]
        candidates = []
        
        for i in range(j):
            d = pj - primes[i]
            states_i = dp[i]
            if not states_i:
                continue
            
            # Find rightmost state with gap <= d
            lo, hi = 0, len(states_i)
            while lo < hi:
                mid = (lo + hi) // 2
                if states_i[mid][0] <= d:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo - 1
            if idx >= 0:
                prev_gap, prev_len, _, _ = states_i[idx]
                candidates.append((d, prev_len + 1, i, idx))
        
        if not candidates:
            dp[j] = [(0, 1, -1, -1)]
            continue
        
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        frontier = []
        max_len_seen = -1
        for gap, length, parent_idx, parent_state_idx in candidates:
            if length > max_len_seen:
                if frontier and frontier[-1][0] == gap:
                    frontier[-1] = (gap, length, parent_idx, parent_state_idx)
                else:
                    frontier.append((gap, length, parent_idx, parent_state_idx))
                max_len_seen = length
        
        dp[j] = frontier
        
        if frontier:
            last_gap = frontier[-1][0]
            last_len = frontier[-1][1]
            if last_len > best_len:
                best_len = last_len
                best_end = j
                best_gap = last_gap
                best_state_idx = len(frontier) - 1
    
    # Reconstruct
    seq = [primes[best_end]]
    cur_idx = best_end
    cur_state_idx = best_state_idx
    
    while True:
        gap, length, parent_idx, parent_state_idx = dp[cur_idx][cur_state_idx]
        if parent_idx == -1:
            break
        seq.append(primes[parent_idx])
        cur_idx = parent_idx
        cur_state_idx = parent_state_idx
    
    seq.reverse()
    
    # Verify
    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        valid = all(gaps[k+1] >= gaps[k] for k in range(len(gaps)-1))
        if not valid:
            values = [primes[0]]
            for p in primes[1:]:
                if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                    values.append(p)
            return values
    
    return seq

# EVOLVE-BLOCK-END
