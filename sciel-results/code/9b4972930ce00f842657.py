import time
import math

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin

    N_LIMIT = 10000
    
    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (N_LIMIT + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for p in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = bytearray(len(is_prime[p*p::p]))
            
    primes = [i for i, val in enumerate(is_prime) if val]
    n = len(primes)
    
    if n < 2:
        return primes
        
    # dp[i][j] = length of longest valid subsequence ending with primes[j], primes[i]
    # where j < i
    # We store prev[i][j] to reconstruct the path
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    max_len = 0
    best_i = -1
    best_j = -1
    
    # We process j from 0 to n-1
    # For each j, we compute dp[i][j] for all i > j
    # To do this efficiently, we need RMQ on dp[j][0..j-1]
    
    for j in range(n):
        # Build sparse table for row j (indices 0 to j-1)
        # If j == 0, no previous elements, so dp[i][0] = 2 for all i > 0?
        # Wait, if j=0, there is no k < 0. So dp[i][0] = 2.
        # If j > 0, we have values dp[j][0], ..., dp[j][j-1].
        
        if j == 0:
            for i in range(1, n):
                dp[i][0] = 2
                prev[i][0] = -1 # No previous element before primes[0] in the subsequence
                if dp[i][0] > max_len:
                    max_len = dp[i][0]
                    best_i = i
                    best_j = 0
        else:
            # Build sparse table for array A where A[k] = dp[j][k] for k in 0..j-1
            # Size of A is j
            size = j
            log_size = int(math.log2(size)) + 1 if size > 0 else 0
            # sparse_table[k][i] = max of A[i .. i + 2^k - 1]
            # We only need to query ranges [k_min, j-1]
            # Simpler: just use a list of prefix maxes? No, range is arbitrary [k_min, j-1].
            # Since j is small (max 1229), we can just iterate k from j-1 down to 0?
            # Cost per i is O(j). Total cost O(n^3) = 1229^3 ~ 1.8 * 10^9. Too slow.
            
            # Let's use Sparse Table properly.
            # K = floor(log2(size)) + 1
            K = 1
            while (1 << K) <= size:
                K += 1
            st = [[0] * size for _ in range(K)]
            
            # Base case
            for k in range(size):
                st[0][k] = dp[j][k]
                
            for p in range(1, K):
                for k in range(size - (1 << p) + 1):
                    st[p][k] = max(st[p-1][k], st[p-1][k + (1 << (p-1))])
            
            def rmq(l, r):
                # max of A[l..r] inclusive
                if l > r:
                    return 0
                length = r - l + 1
                k = int(math.log2(length))
                return max(st[k][l], st[k][r - (1 << k) + 1])
            
            for i in range(j + 1, n):
                gap = primes[i] - primes[j]
                # Find k_min such that primes[k] >= primes[j] - gap
                target = primes[j] - gap
                # Binary search in primes[0..j-1]
                # We need the smallest index k in [0, j-1] such that primes[k] >= target
                # If no such k, k_min = j (meaning empty range)
                
                lo, hi = 0, j - 1
                k_min = j
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if primes[mid] >= target:
                        k_min = mid
                        hi = mid - 1
                    else:
                        lo = mid + 1
                
                if k_min < j:
                    max_val = rmq(k_min, j - 1)
                    dp[i][j] = 1 + max_val
                    # Find which k gave the max value to store in prev
                    # We need to find k in [k_min, j-1] such that dp[j][k] == max_val
                    # And we want the one that allows reconstruction.
                    # Any such k works. Let's find the first one.
                    for k in range(k_min, j):
                        if dp[j][k] == max_val:
                            prev[i][j] = k
                            break
                else:
                    dp[i][j] = 2
                    prev[i][j] = -1
                    
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    best_i = i
                    best_j = j
                    
        if time.time() - t0 > DEADLINE:
            # Return best found so far
            break

    # Reconstruct sequence
    if best_i == -1:
        return primes[:1]
        
    seq = [primes[best_i], primes[best_j]]
    cur_i = best_i
    cur_j = best_j
    while prev[cur_i][cur_j] != -1:
        cur_k = prev[cur_i][cur_j]
        seq.append(primes[cur_k])
        cur_i, cur_j = cur_j, cur_k
        
    seq.reverse()
    return seq
