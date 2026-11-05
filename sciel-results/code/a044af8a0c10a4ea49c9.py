import time
from math import isqrt
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)
    
    # dp[i] is a list of tuples (gap, length, prev_prime_idx, prev_idx_in_dp)
    # sorted by gap, with strictly increasing lengths.
    dp = [[] for _ in range(n)]
    
    # Helper to get max length in dp[j] for gap <= g
    # dp[j] is sorted by gap. We binary search for the rightmost gap <= g.
    def get_max_len(j, g):
        if not dp[j]:
            return 0, -1
        # dp[j] is list of (gap, len, prev, idx)
        # We want the last index k such that dp[j][k][0] <= g
        # Using bisect on a list of gaps
        # To avoid creating a list of gaps every time, we can just binary search manually or store gaps separately.
        # Given small N, manual binary search or bisect on a precomputed list is fine.
        # Let's precompute gaps_list for each j? No, dp[j] changes as we build?
        # No, dp[j] is finalized when we move to i > j.
        # So we can precompute gaps_list[j] after dp[j] is built.
        
        # For now, simple binary search on dp[j]
        lo, hi = 0, len(dp[j]) - 1
        res_idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if dp[j][mid][0] <= g:
                res_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if res_idx == -1:
            return 0, -1
        return dp[j][res_idx][1], res_idx

    for i in range(n):
        if time.time() > deadline:
            # Fallback to simple greedy if DP is too slow (unlikely)
            break
            
        candidates = []
        # Base case: sequence of length 1 ending at primes[i]
        # Gap 0, Len 1, Prev -1, PrevIdx -1
        candidates.append((0, 1, -1, -1))
        
        for j in range(i):
            g = primes[i] - primes[j]
            # We need a sequence ending at j with last gap <= g
            # If dp[j] is empty, skip? No, dp[j] always has at least the base case (0,1,-1,-1)
            prev_len, prev_idx = get_max_len(j, g)
            
            if prev_len > 0:
                new_len = prev_len + 1
                # Candidate: gap g, len new_len, prev prime j, prev idx in dp[j]
                candidates.append((g, new_len, j, prev_idx))
        
        # Sort candidates by gap
        candidates.sort(key=lambda x: x[0])
        
        # Prune to keep Pareto optimal (max length for min gap)
        # We want to keep states (g, L) such that no other state (g', L') has g' <= g and L' >= L
        # Since sorted by g, we just need L to be strictly increasing.
        pruned = []
        max_len_so_far = 0
        for c in candidates:
            g, L, j, idx = c
            if L > max_len_so_far:
                pruned.append(c)
                max_len_so_far = L
        
        dp[i] = pruned

    # Find the best overall
    best_len = 0
    best_i = -1
    for i in range(n):
        if dp[i]:
            # The best state in dp[i] is the last one (max length)
            L = dp[i][-1][1]
            if L > best_len:
                best_len = L
                best_i = i

    # Reconstruct
    if best_i == -1:
        return [primes[0]] if n > 0 else []

    seq_indices = []
    curr_i = best_i
    while curr_i != -1:
        seq_indices.append(curr_i)
        # Get the best state for curr_i
        # It's the last one in dp[curr_i]
        state = dp[curr_i][-1]
        # state is (gap, len, prev_j, prev_idx)
        prev_j = state[2]
        curr_i = prev_j
        
    seq_indices.reverse()
    
    return [primes[i] for i in seq_indices]
