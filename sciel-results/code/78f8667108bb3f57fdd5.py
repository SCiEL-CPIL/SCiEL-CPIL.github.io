import math
import time

def construct():
    target = 123457
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # Precompute largest prime factor for all numbers up to target using sieve
    lpf = [0] * (target + 1)
    lpf[1] = 1
    for i in range(2, target + 1):
        if lpf[i] == 0:
            for j in range(i, target + 1, i):
                lpf[j] = i

    # Now search for the optimal split
    # We want to minimize max(P(a), P(b)) where a + b = target
    # Strategy: try increasing B values and check if there exists a split where both have LPF <= B
    
    best_max_p = float('inf')
    best_a = 1
    best_b = target - 1

    # Generate list of primes up to target for potential B values
    # But we can be smarter: just check if for a given B, there's an a such that
    # lpf[a] <= B and lpf[target-a] <= B
    
    # First, let's find what the minimum possible B is by checking all splits
    half = target // 2
    
    # To speed this up, we can group numbers by their LPF
    # But a simpler approach: just scan all a and track the minimum of max(lpf[a], lpf[target-a])
    
    for a in range(1, half + 1):
        b = target - a
        pa = lpf[a]
        pb = lpf[b]
        cur_max = pa if pa > pb else pb
        if cur_max < best_max_p:
            best_max_p = cur_max
            best_a = a
            best_b = b
            if best_max_p <= 2:
                break
        if time.time() > deadline:
            break

    return [best_a, best_b]
