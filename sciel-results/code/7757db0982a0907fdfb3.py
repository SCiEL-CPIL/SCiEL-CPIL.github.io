import numpy as np

def construct():
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    G = limit
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # M[i, g] = max length of valid sequence ending at primes[i] with last gap <= g
    # Initialize to 1 (sequence of length 1)
    M = np.ones((N, G + 1), dtype=np.int16)
    
    # DP
    for j in range(N):
        pj = primes[j]
        # cand_arr[g] will hold the best length ending at j with EXACTLY gap g 
        # derived from a previous prime, or 1 if no such prime exists (base case)
        # We initialize to 1 because a sequence of length 1 is valid for any gap constraint >= 0
        cand_arr = np.ones(G + 1, dtype=np.int16)
        
        # Iterate over all previous primes
        # We can optimize by only checking i where pj - primes[i] is a valid gap
        # But simply looping i is O(N^2) total, which is fine.
        for i in range(j):
            pi = primes[i]
            g = pj - pi
            # g is always >= 1 and <= G
            # The previous state must allow gap <= g
            # So we look up M[i, g]
            val = 1 + M[i, g]
            # Since pi is unique for each g for a fixed pj, we just assign
            # Note: multiple i could yield same g? No, pi = pj - g is unique.
            # So no conflict, but if there were, we'd take max.
            # Here it's a direct assignment.
            cand_arr[g] = val
            
        # M[j, :] is the prefix max of cand_arr
        # This ensures M[j, g] >= M[j, g-1]
        M[j, :] = np.maximum.accumulate(cand_arr)
        
    # Find max length
    max_len = np.max(M)
    
    # Reconstruct
    flat_idx = np.argmax(M)
    j = flat_idx // (G + 1)
    g = flat_idx % (G + 1)
    
    seq = []
    while True:
        seq.append(primes[j])
        if M[j, g] == 1:
            break
            
        # Find the actual last gap used
        # M[j, g] is the max length with gap <= g.
        # We need the smallest g_last such that M[j, g_last] == M[j, g]
        g_last = g
        while g_last > 0 and M[j, g_last - 1] == M[j, g]:
            g_last -= 1
            
        # The gap used was g_last
        # Previous prime is primes[j] - g_last
        prev_p = primes[j] - g_last
        # prev_p must be a prime
        i = prime_to_idx[prev_p]
        
        j = i
        g = g_last
        
    seq.reverse()
    return seq
