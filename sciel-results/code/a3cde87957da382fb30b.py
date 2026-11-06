import time

def construct():
    deadline = time.time() + 50.0
    N_MAX = 1000000
    N_MIN = 10000
    limit = N_MAX + 1  # We need lpf up to N_MAX + 1
    
    # 1. Compute Largest Prime Factor (LPF) for each number up to limit
    # lpf[x] will store the largest prime factor of x
    # Initialize with 0
    lpf = [0] * (limit + 1)
    
    # Sieve: Iterate i from 2 to limit
    # If lpf[i] is 0, i is prime
    # For each prime i, update lpf for all its multiples
    # Since i increases, the last prime divisor found is the largest
    check_interval = 10000
    for i in range(2, limit + 1):
        if lpf[i] == 0:  # i is prime
            # Update multiples of i
            # j takes values i, 2i, 3i, ...
            for j in range(i, limit + 1, i):
                lpf[j] = i
        if i % check_interval == 0 and time.time() > deadline:
            # Safety break
            break

    # 2. Find n in [N_MIN, N_MAX] that minimizes max(lpf[n], lpf[n+1])
    best_n = N_MIN
    # Initialize best_score with the score of the first candidate
    best_score = max(lpf[N_MIN], lpf[N_MIN + 1])
    
    check_interval_scan = 10000
    for n in range(N_MIN + 1, N_MAX + 1):
        # The largest prime factor of n(n+1) is max(lpf[n], lpf[n+1])
        current_score = max(lpf[n], lpf[n + 1])
        
        if current_score < best_score:
            best_score = current_score
            best_n = n
            
        if (n - N_MIN) % check_interval_scan == 0 and time.time() > deadline:
            break
    
    return best_n
