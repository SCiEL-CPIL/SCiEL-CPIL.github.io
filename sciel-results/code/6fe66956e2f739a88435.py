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
    
    for j in range(n):
        if j == 0:
            for i in range(1, n):
                dp[i][0] = 2
                prev[i][0] = -1
                if dp[i][0] > max_len:
                    max_len = dp[i][0]
                    best_i = i
                    best_j = 0
        else:
            # Build suffix max array for dp[j][0..j-1]
            # suffix_max[k] = max(dp[j][k], dp[j][k+1], ..., dp[j][j-1])
            suffix_max = [0] * (j + 1)
            suffix_max[j] = 0
            for k in range(j - 1, -1, -1):
                suffix_max[k] = max(dp[j][k], suffix_max[k + 1])
            
            # argmax_suffix[k] = index in [k, j-1] achieving suffix_max[k]
            argmax_suffix = [0] * (j + 1)
            argmax_suffix[j] = -1
            for k in range(j - 1, -1, -1):
                if dp[j][k] >= suffix_max[k + 1]:
                    argmax_suffix[k] = k
                else:
                    argmax_suffix[k] = argmax_suffix[k + 1]
            
            # Two-pointer: as i increases, target decreases, so k_min is non-increasing
            # k_min starts at j (no valid k) and decreases as target decreases
            k_min = j
            for i in range(j + 1, n):
                target = 2 * primes[j] - primes[i]
                
                # Decrease k_min while primes[k_min-1] >= target
                while k_min > 0 and primes[k_min - 1] >= target:
                    k_min -= 1
                
                if k_min < j:
                    max_val = suffix_max[k_min]
                    dp[i][j] = 1 + max_val
                    prev[i][j] = argmax_suffix[k_min]
                else:
                    dp[i][j] = 2
                    prev[i][j] = -1
                    
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    best_i = i
                    best_j = j
                    
        if time.time() - t0 > DEADLINE:
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
