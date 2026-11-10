"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import bisect
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
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
    
    primes_arr = np.array(primes, dtype=np.int32)
    
    # DP arrays: for each prime index j, store the Pareto frontier of (gap, length, parent_info)
    # We use lists of tuples for flexibility
    dp_gaps = [[] for _ in range(m)]
    dp_lens = [[] for _ in range(m)]
    dp_parents = [[] for _ in range(m)]
    
    # Base case
    dp_gaps[0] = [0]
    dp_lens[0] = [1]
    dp_parents[0] = [(-1, 0)]
    
    best_len = 1
    best_end = 0
    best_gap = 0
    
    for j in range(1, m):
        if time.time() - t0 > DEADLINE:
            break
        
        pj = primes[j]
        
        # For all i < j, compute gap d = pj - primes[i]
        # We need to find, for each i, the best state in dp[i] with gap <= d
        # Then build the Pareto frontier for j
        
        # Collect candidates: (gap, length, prev_i, prev_gap)
        candidates = []
        
        # Vectorized approach: compute all gaps at once
        gaps_to_j = pj - primes_arr[:j]  # shape (j,)
        
        for i in range(j):
            d = int(gaps_to_j[i])
            gaps_i = dp_gaps[i]
            if not gaps_i:
                continue
            
            # Find best state in i with gap <= d
            idx = bisect.bisect_right(gaps_i, d) - 1
            if idx >= 0:
                prev_gap = gaps_i[idx]
                prev_len = dp_lens[i][idx]
                new_len = prev_len + 1
                candidates.append((d, new_len, i, prev_gap))
        
        if not candidates:
            dp_gaps[j] = [0]
            dp_lens[j] = [1]
            dp_parents[j] = [(-1, 0)]
            continue
        
        # Build Pareto frontier
        # Sort by gap ascending; for same gap, keep max length
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        frontier_gaps = []
        frontier_lens = []
        frontier_parents = []
        max_len = -1
        
        for gap, length, prev_i, prev_gap in candidates:
            if length > max_len:
                frontier_gaps.append(gap)
                frontier_lens.append(length)
                frontier_parents.append((prev_i, prev_gap))
                max_len = length
        
        dp_gaps[j] = frontier_gaps
        dp_lens[j] = frontier_lens
        dp_parents[j] = frontier_parents
        
        if frontier_lens:
            last_gap = frontier_gaps[-1]
            last_len = frontier_lens[-1]
            if last_len > best_len:
                best_len = last_len
                best_end = j
                best_gap = last_gap
    
    # Reconstruct
    seq = [primes[best_end]]
    cur_idx = best_end
    cur_gap = best_gap
    
    idx_in_frontier = bisect.bisect_left(dp_gaps[cur_idx], cur_gap)
    
    if idx_in_frontier < len(dp_gaps[cur_idx]) and dp_gaps[cur_idx][idx_in_frontier] == cur_gap:
        while True:
            prev_i, prev_gap = dp_parents[cur_idx][idx_in_frontier]
            if prev_i == -1:
                break
            seq.append(primes[prev_i])
            cur_idx = prev_i
            cur_gap = prev_gap
            idx_in_frontier = bisect.bisect_left(dp_gaps[cur_idx], cur_gap)
            if idx_in_frontier >= len(dp_gaps[cur_idx]) or dp_gaps[cur_idx][idx_in_frontier] != cur_gap:
                break
    else:
        pass

    seq.reverse()
    
    # Verify
    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        for k in range(len(gaps)-1):
            if gaps[k+1] < gaps[k]:
                values = [primes[0]]
                for p in primes[1:]:
                    if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                        values.append(p)
                return values
    
    return seq

# EVOLVE-BLOCK-END
