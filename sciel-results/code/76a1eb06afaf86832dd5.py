"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 45  # safety margin

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
    # Condition: primes[j] - primes[i] <= primes[k] - primes[j]
    #            i.e., primes[i] >= 2*primes[j] - primes[k]
    
    # For fixed j, we need for each k > j:
    #   max{dp[i][j] : i < j and primes[i] >= 2*primes[j] - primes[k]}
    # Since primes is sorted, primes[i] >= threshold means i >= lo where lo = lower_bound.
    # As k increases, threshold decreases, so lo decreases (moves left).
    # We use suffix-max: suff_val[lo] = max{dp[i][j] : i in [lo, j-1]}
    
    dp = np.zeros((n, n), dtype=np.int32)
    prev = np.full((n, n), -1, dtype=np.int32)

    best_len = 2
    best_j, best_k = 0, 1

    for j in range(1, n):
        if time.time() > deadline:
            break
        
        # Build suffix-max arrays for dp[i][j] over i in [0, j-1]
        # suff_val[i] = max{dp[t][j] : t in [i, j-1]}
        # suff_arg[i] = argmax
        suff_val = np.zeros(j + 1, dtype=np.int32)
        suff_arg = np.full(j + 1, -1, dtype=np.int32)
        
        for i in range(j - 1, -1, -1):
            if dp[i, j] >= suff_val[i + 1]:
                suff_val[i] = dp[i, j]
                suff_arg[i] = i
            else:
                suff_val[i] = suff_val[i + 1]
                suff_arg[i] = suff_arg[i + 1]

        # Two-pointer: lo = smallest index in [0, j-1] with primes[lo] >= threshold
        # As k increases, threshold decreases, so lo can only decrease (move left)
        lo = j - 1  # Start at j-1 for first k
        for k in range(j + 1, n):
            threshold = 2 * primes[j] - primes[k]
            
            # Move lo left while primes[lo-1] >= threshold
            while lo > 0 and primes[lo - 1] >= threshold:
                lo -= 1
            
            if lo >= j:
                dp[j, k] = 2
                prev[j, k] = -1
            else:
                bv = suff_val[lo]
                ba = suff_arg[lo]
                if bv > 0:
                    dp[j, k] = bv + 1
                    prev[j, k] = ba
                else:
                    dp[j, k] = 2
                    prev[j, k] = -1

            if dp[j, k] > best_len:
                best_len = dp[j, k]
                best_j, best_k = j, k

    if best_len <= 2:
        return primes[:2]

    seq = [primes[best_k], primes[best_j]]
    j, k = best_j, best_k
    while len(seq) < best_len:
        i = int(prev[j, k])
        if i < 0:
            break
        seq.append(primes[i])
        k = j
        j = i

    seq.reverse()
    return seq
# EVOLVE-BLOCK-END
