import time

def construct():
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    N = 1000001  # max value is 1000000, need lpf up to 1000001
    
    # Sieve for largest prime factor
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    for n in range(10000, 1000001):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
