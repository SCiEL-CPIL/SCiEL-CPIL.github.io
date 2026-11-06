"""Combined DP: Multi-state Pareto frontier with binary search optimization."""
# EVOLVE-BLOCK-START
import time
import bisect
from math import isqrt

def _sieve_primes(limit):
    """Return list of primes <= limit."""
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i, v in enumerate(is_prime) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill

    primes = _sieve_primes(10000)
    n = len(primes)
    if n == 0:
        return []
    if n == 1:
        return primes

    # dp[i] is a list of states: (last_gap, length, prev_idx, prev_state_idx)
    # States are sorted by last_gap ascending.
    # prev_state_idx is the index in the PRUNED dp[prev_idx] list.
    dp = []
    # For each dp[i], store the gaps separately for faster binary search
    dp_gaps = []

    for i in range(n):
        if time.time() > deadline:
            break
            
        candidates = []
        # Base case: sequence of length 1 ending at primes[i]
        # last_gap = 0, length = 1, no prev
        candidates.append((0, 1, -1, -1))
        
        pi = primes[i]
        for j in range(i):
            gap = pi - primes[j]
            if not dp[j]:
                continue
            
            # Binary search for the rightmost gap <= current gap
            gaps_j = dp_gaps[j]
            k = bisect.bisect_right(gaps_j, gap) - 1
            if k >= 0:
                # In our pruned list, lengths are strictly increasing with gap
                # So the state at index k has the max length for prefix 0..k
                best_len_j = dp[j][k][1]
                new_len = best_len_j + 1
                candidates.append((gap, new_len, j, k))
        
        # Prune candidates: sort by gap, keep only strictly increasing lengths
        candidates.sort(key=lambda x: x[0])
        
        pruned = []
        max_len_so_far = 0
        for c in candidates:
            gap, length, pj, ps = c
            if length > max_len_so_far:
                pruned.append(c)
                max_len_so_far = length
        
        dp.append(pruned)
        dp_gaps.append([s[0] for s in pruned])

    # Find the best sequence
    best_len = 0
    best_i = -1
    best_k = -1
    
    for i in range(len(dp)):
        if dp[i]:
            last_state = dp[i][-1]
            if last_state[1] > best_len:
                best_len = last_state[1]
                best_i = i
                best_k = len(dp[i]) - 1

    if best_i == -1:
        return []
        
    # Reconstruct by following parent pointers
    seq_indices = []
    curr_i = best_i
    curr_k = best_k
    while curr_i != -1:
        seq_indices.append(curr_i)
        gap, length, pj, ps = dp[curr_i][curr_k]
        curr_i = pj
        curr_k = ps
        
    seq_indices.reverse()
    result = [primes[idx] for idx in seq_indices]
    
    return result
# EVOLVE-BLOCK-END
