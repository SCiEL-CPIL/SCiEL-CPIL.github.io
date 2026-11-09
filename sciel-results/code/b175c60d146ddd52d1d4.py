import time
import sys

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin for 90s limit
    
    N = 1000000
    WINDOW = 24
    MAX_VAL = N + WINDOW
    
    # Primes strictly below 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Step 1: Compute S(m) for m in [1, MAX_VAL]
    # S(m) is the part of m composed of primes < 24
    S = [0] * (MAX_VAL + 1)
    S[1] = 1
    
    # Sieve-like approach to compute S(m)
    # We can use the property:
    # Let spf[m] be the smallest prime factor of m.
    # If spf[m] < 24, S[m] = S[m // spf[m]] * spf[m]
    # Else S[m] = S[m // spf[m]]
    
    # Compute smallest prime factor (spf) up to MAX_VAL
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    for m in range(2, MAX_VAL + 1):
        p = spf[m]
        if p < 24:
            S[m] = S[m // p] * p
        else:
            S[m] = S[m // p]
            
    # Step 2: Sliding window to find min distinct count
    # We want to minimize the number of distinct values in S[n+1 : n+25]
    
    # Initialize frequency dictionary for the first window [1001, 1024]
    freq = {}
    distinct = 0
    
    def add_val(v):
        nonlocal distinct
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
            distinct += 1
            
    def remove_val(v):
        nonlocal distinct
        freq[v] -= 1
        if freq[v] == 0:
            del freq[v]
            distinct -= 1
            
    # Initial window: n=1000, values S[1001]...S[1024]
    for i in range(1001, 1025):
        add_val(S[i])
        
    best_n = 1000
    best_distinct = distinct
    
    # Slide window
    # n goes from 1001 to N
    # When n increases by 1, we remove S[n] and add S[n+24]
    # Window for n is S[n+1]...S[n+24]
    # Window for n-1 was S[n]...S[n+23]
    # So remove S[n], add S[n+24]
    
    for n in range(1001, N + 1):
        if time.time() > deadline - 2:
            break
            
        # Remove S[n]
        remove_val(S[n])
        # Add S[n+24]
        add_val(S[n+24])
        
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            # If we find 1, it's the absolute minimum (though impossible as discussed, 
            # but 2 or 3 would be great)
            if best_distinct <= 2:
                # We can break early if we find a very low number, 
                # but let's run a bit more to be sure if time permits.
                # Actually, if we find 2, it's hard to beat.
                # Let's continue to see if we find 1 (impossible) or just return if time is tight.
                pass
                
    return best_n

# EVOLVE-BLOCK-END
