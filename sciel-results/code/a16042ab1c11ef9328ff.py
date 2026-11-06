import numpy as np

def construct():
    # Generate primes up to 10000
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
    
    # Map prime value to index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # M[i, g] = max length of valid sequence ending at primes[i] with last gap <= g
    # Initialize with 1 (sequence of length 1 is always valid)
    M = np.ones((N, G + 1), dtype=np.int16)
    
    for j in range(N):
        pj = primes[j]
        
        # Create Cand array
        # Cand[g] is the candidate length if we use gap g to reach j.
        # If pj - g is prime i, Cand[g] = 1 + M[i, g]. Else 0.
        Cand = np.zeros(G + 1, dtype=np.int16)
        Cand[0] = 1 # Base case for prefix max
        
        # Iterate over all previous primes i < j
        for i in range(j):
            pi = primes[i]
            g = pj - pi
            # g is always <= G because pj <= 10000 and pi >= 2
            val = 1 + M[i, g]
            if val > Cand[g]:
                Cand[g] = val
                
        M[j, :] = np.maximum.accumulate(Cand)
        
    # Find the maximum length
    max_len = np.max(M)
    
    # Reconstruct the sequence
    flat_idx = np.argmax(M)
    j_end = flat_idx // (G + 1)
    g_end = flat_idx % (G + 1)
    
    # Backtrack
    seq = []
    j = j_end
    g = g_end
    while True:
        seq.append(primes[j])
        if M[j, g] == 1:
            break
        
        # Find the smallest gap g' <= g such that M[j, g'] == M[j, g]
        # This g' is the gap actually used in the optimal sequence for state (j, g)
        while g > 0 and M[j, g] == M[j, g - 1]:
            g -= 1
            
        if g == 0:
            break # Should not happen for length > 1
            
        # The transition used gap g.
        # Previous prime is primes[j] - g.
        prev_prime_val = primes[j] - g
        if prev_prime_val < 2 or not is_prime[prev_prime_val]:
             # This should not happen if logic is correct
             break
             
        i = prime_to_idx[prev_prime_val]
        
        # Now find g_prev for i
        # The value used was M[i, g] (max over g' <= g).
        # We need the smallest g_prev <= g such that M[i, g_prev] == M[i, g].
        g_prev = g
        while g_prev > 0 and M[i, g_prev - 1] == M[i, g]:
            g_prev -= 1
            
        j = i
        g = g_prev
        
    seq.reverse()
    return seq

# Test
# print(construct())
