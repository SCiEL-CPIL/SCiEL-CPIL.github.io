import time
import math
from bisect import bisect_left, bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # Sieve of Eratosthenes to get primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)  # ~1229 primes
    
    # Compute consecutive prime gaps
    m = n - 1  # number of gaps
    gaps = [primes[i + 1] - primes[i] for i in range(m)]
    
    # Precompute prefix sums of gaps
    prefix = [0] * (m + 1)
    for i in range(m):
        prefix[i + 1] = prefix[i] + gaps[i]
    
    # dp[end] = list of (length, last_segment_sum, prev_end, prev_state_idx)
    # The list is sorted by length desc, and last_segment_sum is strictly decreasing
    dp = [[] for _ in range(m)]
    
    # Base case: a single segment from 0 to end
    for end in range(m):
        seg_sum = prefix[end + 1] - prefix[0]
        dp[end].append((1, seg_sum, -1, -1))
        
    best_len = 1
    best_end = 0
    best_state_idx = 0
    
    for end in range(1, m):
        if time.time() > deadline:
            break
            
        candidates = [(1, prefix[end + 1], -1, -1)]
        
        for prev_end in range(end):
            next_seg_sum = prefix[end + 1] - prefix[prev_end + 1]
            states = dp[prev_end]
            if not states:
                continue
            
            # Binary search for the best state in `states` with S <= next_seg_sum
            # states are sorted by L desc, S strictly decreasing
            # We want the smallest index k such that states[k][1] <= next_seg_sum
            lo = 0
            hi = len(states)
            while lo < hi:
                mid = (lo + hi) // 2
                if states[mid][1] <= next_seg_sum:
                    hi = mid
                else:
                    lo = mid + 1
            
            if lo < len(states):
                L, S, pe, si = states[lo]
                candidates.append((L + 1, next_seg_sum, prev_end, lo))
        
        # Prune candidates
        # Sort by length desc, then sum asc
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        non_dom = []
        min_sum_encountered = float('inf')
        for cand in candidates:
            L, S, pe, si = cand
            if S < min_sum_encountered:
                non_dom.append(cand)
                min_sum_encountered = S
        
        dp[end] = non_dom
        
        if non_dom:
            current_best = non_dom[0][0]
            if current_best > best_len:
                best_len = current_best
                best_end = end
                best_state_idx = 0
    
    # Reconstruct
    segment_ends = []
    end = best_end
    state_idx = best_state_idx
    while end != -1:
        L, S, prev_end, prev_state_idx = dp[end][state_idx]
        segment_ends.append(end)
        end = prev_end
        state_idx = prev_state_idx
    
    segment_ends.reverse()
    
    prime_indices = [0]
    for se in segment_ends:
        prime_indices.append(se + 1)
    
    result = [primes[i] for i in prime_indices]
    return result
