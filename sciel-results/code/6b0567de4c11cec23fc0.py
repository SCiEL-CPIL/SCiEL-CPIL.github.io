import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 1000000
    # Sieve: lpf[i] = largest prime factor of i
    lpf = [1] * (N + 2)
    for p in range(2, N + 1):
        if lpf[p] == 1:  # p is prime
            for m in range(p, N + 1, p):
                lpf[m] = p

    # Scan for best n in [10000, 1000000]
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])

    for n in range(10001, N + 1):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n

    return best_n
