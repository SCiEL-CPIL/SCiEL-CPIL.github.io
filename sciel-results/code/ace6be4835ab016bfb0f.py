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
        # Precompute i_for_g for this j
        # i_for_g[g] = i if pj - g is a prime, else -1
        i_for_g = np.full(G + 1, -1, dtype=np.int16)
        # Only g such that pj - g >= 2
        max_g = pj - 2
        if max_g > 0:
            # Vectorized check for primes
            # vals = pj - np.arange(1, max_g + 1)
            # This creates an array of values to check
            # We can just loop g, it's fast enough
            for g in range(1, max_g + 1):
                val = pj - g
                if is_prime[val]:
                    i_for_g[g] = prime_to_idx[val]
        
        # Compute M[j, :]
        # M[j, 0] = 1 (gap 0 is not used, but prefix max starts here)
        # Actually, gaps are >= 2 for distinct primes.
        # M[j, g] for g < 2 should be 1.
        # The loop starts at g=1.
        for g in range(1, G + 1):
            i = i_for_g[g]
            if i != -1:
                # Candidate length
                cand = 1 + M[i, g]
                # Prefix max
                if cand > M[j, g - 1]:
                    M[j, g] = cand
                else:
                    M[j, g] = M[j, g - 1]
            else:
                M[j, g] = M[j, g - 1]
                
    # Find the maximum length
    max_len = np.max(M)
    
    # Reconstruct the sequence
    # We need to find the j and g that give max_len
    # There might be multiple, pick any.
    # Actually, we want the sequence itself.
    # We need to backtrack.
    # Let's store the parent pointers?
    # That would double memory.
    # Alternatively, we can just find the j and g, and then backtrack using M.
    
    # Find index of max value
    # M is 2D.
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
        # Find previous j and g
        # M[j, g] = 1 + M[i, g_prev] where g_prev <= g
        # Actually, M[j, g] comes from some i and g' such that g' <= g
        # and the transition was from (i, g') to (j, g) with gap g.
        # Wait, the transition is:
        # We are at (j, g). The previous state was (i, g') with g' <= g.
        # And the gap used to get to j was g.
        # So P_j - P_i = g.
        # i is determined by g: P_i = P_j - g.
        # But which g' was used?
        # M[i, g'] was the value used.
        # We need to find the g' that maximized M[i, g'] subject to g' <= g.
        # Actually, M[i, g] is the max over g' <= g.
        # So the value used was M[i, g] (if the transition came from i with gap g).
        # Wait, in the DP step:
        # cand = 1 + M[i, g]
        # This assumes the previous gap was <= g.
        # So the previous state is effectively "ending at i with some gap <= g".
        # To backtrack, we need to know which specific gap g' was used at i.
        # We can find it by looking at M[i, g'] for g' <= g.
        # The max value is M[i, g].
        # We need the largest g' <= g such that M[i, g'] == M[i, g]?
        # Not necessarily largest, just one that achieves the max.
        # We can find the first g' from 1 to g such that M[i, g'] == M[i, g].
        # Or the last?
        # It doesn't matter for length, but it determines the sequence.
        
        # Find i
        val = primes[j] - g
        if not is_prime[val]:
            # This should not happen if M[j, g] > 1
            # Unless M[j, g] came from M[j, g-1]
            # If M[j, g] == M[j, g-1], then the max didn't come from a transition with gap g.
            # It came from a smaller gap.
            # So we should backtrack to (j, g-1)?
            # No, M[j, g] is the max length ending at j with gap <= g.
            # If the max was achieved with a smaller gap, M[j, g] == M[j, g-1].
            # So we can just decrement g until we find the gap that was actually used?
            # Or we can store the argmax gap.
            
            # Let's refine backtracking:
            # If M[j, g] == M[j, g-1], then the optimal sequence for gap <= g is the same as for gap <= g-1.
            # So we can set g = g-1 and continue backtracking at j?
            # No, the sequence ends at j. The last gap is <= g.
            # If M[j, g] == M[j, g-1], the last gap in the optimal sequence is <= g-1.
            # So we can just reduce g and look again.
            # But we need to find the actual last gap.
            # Let's find the smallest g_min such that M[j, g_min] == M[j, g].
            # Then the last gap is g_min? No, the last gap is some g' <= g_min.
            # Actually, if M[j, g] > M[j, g-1], then the max was achieved with a transition using gap g.
            # In that case, i is determined by g.
            
            # So:
            while g > 0 and M[j, g] == M[j, g - 1]:
                g -= 1
            # Now M[j, g] > M[j, g-1] (or g==0)
            # This implies the transition used gap g.
            # So i = prime_to_idx[primes[j] - g]
            
            if g == 0:
                break # Should not happen for length > 1
            
            val = primes[j] - g
            if not is_prime[val]:
                # Error in logic?
                break
            
        i = prime_to_idx[primes[j] - g]
        
        # Now find g_prev for i
        # The value used was M[i, g] (max over g' <= g).
        # We need to find a g' <= g such that M[i, g'] == M[i, g].
        # Let's find the smallest such g'.
        g_prev = g
        while g_prev > 0 and M[i, g_prev - 1] == M[i, g]:
            g_prev -= 1
        # Now M[i, g_prev] is the max, and it was first achieved at g_prev.
        # So the last gap of the sequence ending at i was g_prev?
        # Not necessarily, but M[i, g_prev] represents the best sequence ending at i with gap <= g_prev.
        # So we can set j = i, g = g_prev and continue.
        
        j = i
        g = g_prev
        
    seq.reverse()
    return seq

# Test
# print(construct())
