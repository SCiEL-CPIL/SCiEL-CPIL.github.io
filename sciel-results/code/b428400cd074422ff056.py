"""Smooth two-summand decomposition of a prescribed integer."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s to be safe under 90s kill
    target = 123457
    max_n = target - 1  # 123456

    # Compute largest prime factor for all numbers up to max_n
    # Using a sieve-like approach
    lpf = [0] * (max_n + 1)
    lpf[1] = 1

    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_n + 1, i):
                if lpf[j] == 0 or i > lpf[j]:
                    lpf[j] = i

    # Search for best a in [1, target-1]
    best_a = 1
    best_b = target - 1
    best_max = max(lpf[best_a], lpf[best_b])

    for a in range(1, target):
        b = target - a
        pa = lpf[a]
        pb = lpf[b]
        cur_max = pa if pa > pb else pb
        if cur_max < best_max:
            best_max = cur_max
            best_a = a
            best_b = b
            if time.time() > deadline:
                break
        # Early termination: if we found max=2, that's optimal
        if best_max <= 2:
            break

    return [best_a, best_b]
# EVOLVE-BLOCK-END
