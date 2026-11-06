import time

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    target = 123457
    
    # Compute largest prime factor for all numbers up to target-1 (sieve)
    max_n = target - 1
    lpf = [0] * (max_n + 1)
    lpf[1] = 1
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_n + 1, i):
                lpf[j] = i
    
    # Full scan to find the pair minimizing max(P(a), P(b))
    # This is O(N) and guaranteed to find the global optimum
    best_max = float('inf')
    best_a = 1
    
    for a in range(1, target):
        b = target - a
        cur_max = max(lpf[a], lpf[b])
        if cur_max < best_max:
            best_max = cur_max
            best_a = a
            # Early termination: if best_max == 2, that's the absolute minimum
            # (both a and b would be powers of 2, but a+b=123457 is odd, so
            #  one must be odd -> P(odd) >= 3, so min possible is 3)
            if best_max <= 3:
                break
    
    return [best_a, target - best_a]
