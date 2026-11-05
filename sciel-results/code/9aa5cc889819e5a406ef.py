"""Combined approach: Sieve for LPF + targeted smooth search with deadline guard."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    target = 123457

    # Precompute largest prime factor for all numbers up to target using sieve
    lpf = [0] * (target + 1)
    lpf[1] = 1
    for i in range(2, target + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, target + 1, i):
                lpf[j] = i

    # Initialize best with trivial split a=1
    best_a = 1
    best_b = target - 1
    best_score = max(lpf[best_a], lpf[best_b])

    # Full linear scan (fast enough for 123457)
    half_target = target // 2
    for a in range(2, half_target + 1):
        if time.time() > deadline:
            break
        b = target - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b

    return [best_a, best_b]
