"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    import time
    from math import isqrt
    
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s hard kill
    
    # ---- Sieve of primes up to 10000 ----
    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(N + 1) if is_prime[i]]
    K = len(primes)
    
    # ---- Dynamic Programming over pairs of primes ----
    # We want the longest subsequence of primes p_i1 < p_i2 < ... < p_ik
    # such that gaps g_j = p_{i_{j+1}} - p_{i_j} are non-decreasing: g_1 <= g_2 <= ... <= g_{k-1}
    
    # State: dp[i][j] = maximum length of a valid sequence ending with primes[i] as second-to-last
    # and primes[j] as last, where i < j.
    # Actually, let's define it more carefully.
    # Let dp[i][j] be the length of the longest valid prime subsequence ending with ... , primes[i], primes[j]
    # where the gap between primes[i] and primes[j] is primes[j] - primes[i], and all previous gaps are <= this gap.
    # We need i < j.
    
    # Transition: dp[i][j] = 1 + max(dp[h][i] for h < i if primes[i] - primes[h] <= primes[j] - primes[i])
    # Base case: dp[i][j] = 2 for all i < j (sequence of just two primes)
    
    # To optimize, we can iterate j from 0 to K-1, and for each j, iterate i from 0 to j-1.
    # For each (i, j), we need to find max dp[h][i] over h < i with primes[i] - primes[h] <= primes[j] - primes[i].
    
    # Let's use a different approach. 
    # Let f(i, j) = length of longest valid sequence ending with primes[i], primes[j] (i < j).
    # f(i, j) = 2 + max( f(h, i) for h < i, primes[i] - primes[h] <= primes[j] - primes[i] )
    # If no such h exists, f(i, j) = 2.
    
    # We can compute this in O(K^2) if we are clever, or O(K^3) naively.
    # K is about 1229 (number of primes <= 10000).
    # O(K^3) would be ~1.8 * 10^9, which is too slow in Python.
    # O(K^2 log K) is feasible.
    
    # Let's use the following:
    # For a fixed i, as j increases, the gap primes[j] - primes[i] increases.
    # So the condition primes[i] - primes[h] <= primes[j] - primes[i] becomes easier to satisfy.
    # We can precompute for each i a list of (gap, dp[h][i]) for h < i.
    # Then for each j, we need the max dp[h][i] for h with gap <= primes[j] - primes[i].
    
    # Actually, let's re-index.
    # Let dp[i][j] be defined for i < j.
    # We iterate j from 1 to K-1.
    # For each j, we iterate i from 0 to j-1.
    # We want dp[i][j] = 2 + max( dp[h][i] for h in 0..i-1 such that primes[i] - primes[h] <= primes[j] - primes[i] )
    
    # For a fixed i, let's consider all h < i. We have pairs (primes[i] - primes[h], dp[h][i]).
    # As j increases, primes[j] - primes[i] increases.
    # So we can sort the h's by primes[i] - primes[h] (which is the same as sorting by primes[h] descending).
    # And maintain a running maximum of dp[h][i].
    
    # But i changes for each j. So we can't easily precompute across i.
    
    # Alternative: Iterate i from 0 to K-1.
    # For each i, we want to compute dp[i][j] for all j > i.
    # For a fixed i, we have h < i. We sort h by primes[i] - primes[h] ascending.
    # Then for j = i+1, i+2, ..., K-1, primes[j] - primes[i] increases.
    # We can use a pointer to track which h's are valid.
    
    # Let's try this.
    
    # dp[i][j] for i < j. We'll use a 2D array, but K is ~1229, so K^2 is ~1.5M, which is fine.
    
    dp = [[0] * K for _ in range(K)]
    best = 2  # minimum length is 2 (any two primes)
    
    # For each i, we will compute dp[i][j] for j > i.
    # But dp[i][j] depends on dp[h][i] for h < i.
    # So we should iterate i in increasing order? No, dp[h][i] requires h < i, so if we iterate i from 0 to K-1,
    # when we are at i, all dp[h][i] for h < i have already been computed (when we were at h and computed dp[h][i]).
    # Wait, no. dp[h][i] is computed when we process pair (h, i).
    # If we iterate i from 0 to K-1, and for each i iterate j from i+1 to K-1,
    # then when we compute dp[i][j], we need dp[h][i] for h < i.
    # These dp[h][i] were computed when we processed pair (h, i), which happened when we were at h and j=i.
    # Since h < i, and we process i in increasing order, when we are at i, all h < i have been processed.
    # So dp[h][i] is available.
    
    # However, for a fixed i, to compute dp[i][j] for all j > i, we need:
    # dp[i][j] = 2 + max( dp[h][i] for h < i, primes[i] - primes[h] <= primes[j] - primes[i] )
    
    # For fixed i, let's create a list of (gap_h, dp[h][i]) for h in 0..i-1.
    # gap_h = primes[i] - primes[h].
    # Sort this list by gap_h ascending.
    # Then for j = i+1, i+2, ..., K-1:
    #   target_gap = primes[j] - primes[i]
    #   We need max dp[h][i] for gap_h <= target_gap.
    #   Since the list is sorted by gap_h, we can use a pointer.
    
    # But note: as j increases, target_gap increases. So we can use a two-pointer technique.
    
    # Let's implement this.
    
    # Initialize dp with 2 for all valid pairs (i, j) with i < j.
    # Actually, we'll compute it on the fly.
    
    # We need to be careful: dp[i][j] is only defined for i < j.
    
    # Let's use a 2D list dp[i][j] where i < j.
    
    # For i from 0 to K-1:
    #   Create list L_i = [(primes[i] - primes[h], dp[h][i]) for h in range(i)]
    #   Sort L_i by the first element (gap).
    #   For j from i+1 to K-1:
    #       target = primes[j] - primes[i]
    #       Find max dp_val in L_i where gap <= target.
    #       dp[i][j] = 2 + max_dp_val (or 2 if no such h)
    #       Update best.
    
    # But this is O(K^2 log K) due to sorting for each i.
    # Total: sum_{i=0}^{K-1} i log i * (K - i) ~ K^3 log K / 6, which might be too slow.
    
    # Wait, for each i, we sort i elements: O(i log i).
    # Then for each j > i, we do a linear scan or binary search: O(log i) per j.
    # So for fixed i: O(i log i + (K-i) log i) = O(K log i).
    # Total: sum_{i=0}^{K-1} K log i = K * sum_{i=1}^{K-1} log i = K * log(K!) ~ K^2 log K.
    # K ~ 1229, K^2 ~ 1.5M, log K ~ 10, so ~15M operations. This should be fine.
    
    # Let's implement with binary search.
    
    from bisect import bisect_right
    
    for i in range(K):
        if time.time() > deadline:
            break
        if i == 0:
            continue  # no h < 0
        
        # Build list of (gap, dp_val) for h < i
        list_i = []
        for h in range(i):
            gap = primes[i] - primes[h]
            dp_val = dp[h][i]
            if dp_val > 0:  # only consider valid states
                list_i.append((gap, dp_val))
        
        if not list_i:
            # No valid previous states, so dp[i][j] = 2 for all j > i
            for j in range(i + 1, K):
                dp[i][j] = 2
                if 2 > best:
                    best = 2
            continue
        
        # Sort by gap
        list_i.sort(key=lambda x: x[0])
        
        # Extract gaps and dp_vals for binary search
        gaps = [x[0] for x in list_i]
        dp_vals = [x[1] for x in list_i]
        
        # We need prefix max of dp_vals
        prefix_max = [0] * len(dp_vals)
        prefix_max[0] = dp_vals[0]
        for idx in range(1, len(dp_vals)):
            prefix_max[idx] = max(prefix_max[idx - 1], dp_vals[idx])
        
        for j in range(i + 1, K):
            if time.time() > deadline:
                break
            target = primes[j] - primes[i]
            # Find rightmost index where gap <= target
            idx = bisect_right(gaps, target) - 1
            if idx >= 0:
                max_dp = prefix_max[idx]
                dp[i][j] = 2 + max_dp
            else:
                dp[i][j] = 2
            
            if dp[i][j] > best:
                best = dp[i][j]
    
    # Now we need to reconstruct the actual sequence.
    # We have dp[i][j] = length of longest valid sequence ending with primes[i], primes[j].
    # We need to find the pair (i, j) that gives the maximum length, and then backtrack.
    
    max_len = 0
    best_i = -1
    best_j = -1
    for i in range(K):
        for j in range(i + 1, K):
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j
    
    if max_len == 0:
        # Fallback: return first two primes
        return primes[:2]
    
    # Backtrack
    sequence = []
    i, j = best_i, best_j
    sequence.append(primes[j])
    sequence.append(primes[i])
    
    while len(sequence) < max_len:
        # Find h < i such that dp[h][i] = dp[i][j] - 2 and primes[i] - primes[h] <= primes[j] - primes[i]
        target_gap = primes[j] - primes[i]
        found = False
        for h in range(i):
            if dp[h][i] == dp[i][j] - 2:
                if primes[i] - primes[h] <= target_gap:
                    sequence.append(primes[h])
                    i, j = h, i
                    found = True
                    break
        if not found:
            # Should not happen
            break
    
    sequence.reverse()
    
    # Verify the sequence
    # Check that all are primes, increasing, and gaps are non-decreasing
    for k in range(len(sequence) - 1):
        if sequence[k + 1] <= sequence[k]:
            return primes[:2]  # fallback
        if k > 0:
            gap_curr = sequence[k + 1] - sequence[k]
            gap_prev = sequence[k] - sequence[k - 1]
            if gap_curr < gap_prev:
                return primes[:2]  # fallback
    
    return sequence

# EVOLVE-BLOCK-END
