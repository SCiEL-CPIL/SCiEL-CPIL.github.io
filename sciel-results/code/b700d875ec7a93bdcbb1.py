import time
import math

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
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
    # prev[i][j] = index k such that primes[k] is the element before primes[j]
    # If prev[i][j] == -1, the subsequence is just [primes[j], primes[i]]
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    max_len = 0
    best_i = -1
    best_j = -1
    
    for j in range(n):
        if time.time() - t0 > DEADLINE:
            break
            
        if j == 0:
            for i in range(1, n):
                dp[i][0] = 2
                prev[i][0] = -1
                if dp[i][0] > max_len:
                    max_len = dp[i][0]
                    best_i = i
                    best_j = 0
        else:
            # Two-pointer approach: as i increases, threshold decreases,
            # so valid k range only grows. Maintain running max.
            k_ptr = j - 1
            current_max = 0
            best_k = -1
            
            for i in range(j + 1, n):
                threshold = 2 * primes[j] - primes[i]
                
                # Add valid k's to current_max
                # k must satisfy primes[k] >= threshold
                while k_ptr >= 0 and primes[k_ptr] >= threshold:
                    if dp[j][k_ptr] > current_max:
                        current_max = dp[j][k_ptr]
                        best_k = k_ptr
                    k_ptr -= 1
                
                if current_max > 0:
                    dp[i][j] = current_max + 1
                    prev[i][j] = best_k
                else:
                    dp[i][j] = 2
                    prev[i][j] = -1
                    
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    best_i = i
                    best_j = j

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
