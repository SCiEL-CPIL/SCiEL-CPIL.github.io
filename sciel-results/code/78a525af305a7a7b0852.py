import time
from math import isqrt
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    N = 10000
    # Sieve of Eratosthenes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            is_prime[i*i:N+1:i] = [False] * (((N - i*i) // i) + 1)
    
    primes = [i for i, v in enumerate(is_prime) if v]
    K = len(primes)
    
    if K < 2:
        return primes

    # dp[i][j] = length of longest valid subsequence ending with primes[i], primes[j]
    # prev[i][j] = index i' such that the subsequence ending with primes[i'], primes[i] was extended to primes[j]
    # Actually, if state is (i, j), prev should point to the element before i in the sequence.
    # Let's say seq ends with ..., p_i, p_j.
    # The previous state was (p_h, p_i).
    # So prev[i][j] should store h.
    
    dp = [[2] * K for _ in range(K)]
    prev = [[-1] * K for _ in range(K)]
    
    max_len = 2
    best_end_j = 0
    best_end_k = 1 # dummy
    
    # Iterate j from 1 to K-1
    # For each j, we compute dp[j][k] for k > j
    # dp[j][k] depends on dp[i][j] for i < j
    
    for j in range(1, K):
        if time.time() > deadline:
            break
            
        # Compute suffix max and argmax for dp[0..j-1][j]
        # M[i] = max(dp[t][j] for t in [i, j-1])
        # ArgM[i] = t that achieves M[i]
        M = [0] * j
        ArgM = [0] * j
        
        M[j-1] = dp[j-1][j]
        ArgM[j-1] = j-1
        
        for i in range(j-2, -1, -1):
            if dp[i][j] > M[i+1]:
                M[i] = dp[i][j]
                ArgM[i] = i
            else:
                M[i] = M[i+1]
                ArgM[i] = ArgM[i+1]
                
        # Iterate k from j+1 to K-1
        for k in range(j+1, K):
            if time.time() > deadline:
                break
                
            # Condition: p_k - p_j >= p_j - p_i
            # p_i >= 2*p_j - p_k
            L = 2 * primes[j] - primes[k]
            
            # Find smallest i in [0, j) such that primes[i] >= L
            i_min = bisect_left(primes, L, 0, j)
            
            if i_min < j:
                # Valid predecessor exists
                new_len = M[i_min] + 1
                if new_len > dp[j][k]:
                    dp[j][k] = new_len
                    prev[j][k] = ArgM[i_min]
                    
                    if new_len > max_len:
                        max_len = new_len
                        best_end_j = j
                        best_end_k = k

    # Reconstruct sequence
    if max_len == 2:
        # Just return first two primes or any two
        return primes[:2]
        
    seq = [primes[best_end_k], primes[best_end_j]]
    cur_j = best_end_j
    cur_k = best_end_k
    
    while prev[cur_j][cur_k] != -1:
        prev_i = prev[cur_j][cur_k]
        seq.append(primes[prev_i])
        cur_k = cur_j
        cur_j = prev_i
        
    seq.reverse()
    return seq
