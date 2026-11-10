import time

def construct():
    S = 123457
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    # Sieve for largest prime factor
    # Initialize P to 0 to distinguish unvisited (prime) from visited (composite)
    # Note: P[1] will be set to 1 explicitly later
    P = [0] * (S + 1)
    
    for i in range(2, S + 1):
        if P[i] == 0: # i is prime
            for j in range(i, S + 1, i):
                P[j] = i
                
    P[1] = 1
    
    best_score = float('inf')
    best_a = 1
    best_b = S - 1
    
    # Search for optimal split with time guard
    # We only need to check a <= b, so a <= S/2
    half_S = S // 2
    for a in range(1, half_S + 1):
        if time.time() > deadline:
            break
        b = S - a
        pa = P[a]
        pb = P[b]
        score = pa if pa > pb else pb
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
    return [best_a, best_b]
