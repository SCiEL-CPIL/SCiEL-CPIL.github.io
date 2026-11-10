"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = start + 50

    n_max = 10000
    is_prime = [True] * (n_max + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n_max + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, n_max + 1) if is_prime[p]]
    n = len(primes)

    if n <= 2:
        return primes

    # dp[j][k] = longest valid subsequence ending at primes[j], primes[k]
    # prev[j][k] = index i such that subsequence is ...primes[i], primes[j], primes[k]
    # We compute row by row: for each j, we need suffix-max over dp[i][j] for i < j
    # Store only what's needed: dp as list of lists, but we can be smarter
    
    # Key insight: for fixed j, when computing dp[j][k] for all k > j,
    # we need to query: max{dp[i][j] : i < j and primes[i] <= 2*primes[j] - primes[k]}
    # This is a suffix-max over i in [0, j-1] with a threshold on primes[i].
    
    # Since primes are sorted, primes[i] <= threshold means i <= some index.
    # So for fixed j, as k increases, threshold decreases, so the valid i range shrinks.
    
    # We'll use the same suffix-max approach but avoid O(n^2) memory for prev
    # by storing prev as a flat array or recomputing during reconstruction.
    
    # Actually, let's keep the structure but optimize the inner loop.
    
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]

    best_len = 2
    best_j, best_k = 0, 1

    for j in range(1, n):
        if time.time() > deadline:
            break
        
        # Build suffix-max arrays for dp[i][j] over i in [0, j-1]
        # suff_val[i] = max{dp[t][j] : t in [i, j-1]}
        # suff_arg[i] = argmax
        suff_val = [0] * (j + 1)
        suff_arg = [-1] * (j + 1)
        
        # Initialize suff_val[j] = 0, suff_arg[j] = -1 (no valid i >= j)
        # Fill from i = j-1 down to 0
        for i in range(j - 1, -1, -1):
            if dp[i][j] >= suff_val[i + 1]:
                suff_val[i] = dp[i][j]
                suff_arg[i] = i
            else:
                suff_val[i] = suff_val[i + 1]
                suff_arg[i] = suff_arg[i + 1]

        # For each k > j, find the largest i < j such that primes[i] <= 2*primes[j] - primes[k]
        # Since primes is sorted, this is a binary search or two-pointer.
        # As k increases, threshold = 2*primes[j] - primes[k] decreases, so i_max decreases.
        
        lo = j  # Start at j; we'll move left as threshold decreases
        for k in range(j + 1, n):
            threshold = 2 * primes[j] - primes[k]
            
            # Move lo left while primes[lo-1] >= threshold
            # We want the smallest index i such that all t in [i, j-1] have primes[t] >= threshold
            # Actually we want: max over i in [0, j-1] with primes[i] <= threshold
            # Since primes is increasing, this is i <= bisect_right(primes, threshold, 0, j) - 1
            
            # Use two-pointer: lo is the leftmost index in [0, j] such that primes[lo] > threshold
            # So valid i are in [0, lo-1], and we want max dp[i][j] for i in [0, lo-1]
            # That's suff_val[0] if lo > 0, but we need the max over [0, lo-1]
            # Which is suff_val[0] only if lo == j... no.
            
            # Let me re-think. suff_val[i] = max over t in [i, j-1].
            # We want max over i in [0, idx] where idx = largest index with primes[idx] <= threshold.
            # That's not directly suff_val. We need prefix-max, not suffix-max.
            
            # Let me redo: define pref_val[i] = max{dp[t][j] : t in [0, i]} for i in [0, j-1]
            # Then for threshold, find idx = bisect_right(primes, threshold, 0, j) - 1
            # If idx >= 0, answer is pref_val[idx]
            
            # But building prefix-max for each j is O(j), and we do this for each j, so O(n^2) total.
            # That's the same as before.
            
            # Actually the original code uses suffix-max and a two-pointer from the right.
            # Let me stick with that approach but make it correct.
            
            # The condition for a valid transition i -> j -> k is:
            # primes[j] - primes[i] <= primes[k] - primes[j]
            # => primes[i] >= 2*primes[j] - primes[k] = threshold
            # So we need i < j and primes[i] >= threshold.
            # Since primes is sorted, this means i >= some lower bound.
            # Let lo = smallest index in [0, j-1] such that primes[lo] >= threshold.
            # Then valid i are in [lo, j-1], and we want max dp[i][j] over that range.
            # That's exactly suff_val[lo] if lo < j, else no valid i.
            
            # As k increases, threshold decreases, so lo decreases (or stays same).
            # So we can use a two-pointer moving lo leftward.
            
            while lo > 0 and primes[lo - 1] >= threshold:
                lo -= 1
            
            if lo >= j:
                # No valid i
                dp[j][k] = 2
                prev[j][k] = -1
            else:
                bv = suff_val[lo]
                ba = suff_arg[lo]
                if bv > 0:
                    dp[j][k] = bv + 1
                    prev[j][k] = ba
                else:
                    dp[j][k] = 2
                    prev[j][k] = -1

            if dp[j][k] > best_len:
                best_len = dp[j][k]
                best_j, best_k = j, k

    if best_len <= 2:
        return primes[:2]

    seq = [primes[best_k], primes[best_j]]
    j, k = best_j, best_k
    while len(seq) < best_len:
        i = prev[j][k]
        if i < 0:
            break
        seq.append(primes[i])
        k = j
        j = i

    seq.reverse()
    return seq
# EVOLVE-BLOCK-END
