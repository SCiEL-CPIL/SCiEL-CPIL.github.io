import time
import math

def construct():
    # Deadline guard: 60s limit, use 55s to be safe
    start_time = time.time()
    DEADLINE = 55.0
    
    # Range for n is [10000, 1000000]
    # We need largest prime factors for n and n+1
    # Max value is 1000001
    MAX_VAL = 1000001
    
    # Sieve to find largest prime factor for each number up to MAX_VAL
    # Using a list of zeros
    lpf = [0] * (MAX_VAL + 1)
    
    # Standard sieve for Largest Prime Factor
    # i is prime if lpf[i] is 0
    for i in range(2, MAX_VAL + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, MAX_VAL + 1, i):
                lpf[j] = i
                
    # Check time
    if time.time() - start_time > DEADLINE:
        # Fallback: Return a known good candidate or the first one found
        # Since sieve is fast, this is unlikely, but safe.
        # A simple heuristic: 10000 is valid.
        return 10000

    best_n = -1
    best_score = float('inf')
    
    # Iterate through valid n
    # We can stop early if we find a very small prime, but we don't know the min.
    # However, we can just scan all.
    # Optimization: Local variables for speed
    lpf_local = lpf
    best_n_local = -1
    best_score_local = float('inf')
    
    for n in range(10000, MAX_VAL):
        # n+1 is at most 1000001, which is within MAX_VAL
        current_max = lpf_local[n]
        lpf_next = lpf_local[n+1]
        if lpf_next > current_max:
            current_max = lpf_next
        
        if current_max < best_score_local:
            best_score_local = current_max
            best_n_local = n
            
        # Optional: Early exit if we find a very small prime?
        # The minimum possible LPF for n >= 10000 is at least 2.
        # If we find max(LPF(n), LPF(n+1)) == 2, that means both are powers of 2.
        # Consecutive powers of 2? Only 1 and 2. Not in range.
        # So min is > 2.
        
    return best_n_local
