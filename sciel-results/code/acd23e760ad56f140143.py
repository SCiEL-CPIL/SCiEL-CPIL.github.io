import time

def construct():
    S = 123457
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    # Compute largest prime factor for all numbers up to S
    # P[i] will store the largest prime factor of i
    P = [1] * (S + 1)
    
    # Sieve of Eratosthenes variant to find largest prime factor
    for i in range(2, S + 1):
        if P[i] == 1:  # i is prime
            for j in range(i, S + 1, i):
                P[j] = i  # Since we iterate i in increasing order, the last prime 
                          # that divides j is the largest prime factor.
    
    best_score = float('inf')
    best_a = 1
    best_b = S - 1
    
    # Search for optimal split
    # We only need to check a <= S // 2 due to symmetry
    for a in range(1, S // 2 + 1):
        # Time guard
        if time.time() > deadline:
            break
            
        b = S - a
        # Calculate max prime factor for this pair
        pa = P[a]
        pb = P[b]
        score = pa if pa > pb else pb
        
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
    return [best_a, best_b]
