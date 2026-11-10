import time

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin before 60s limit

    limit = 1000000
    max_val = limit + 1

    # Sieve to compute the largest prime factor (LPF) for every number up to max_val.
    # lpf[i] will hold the largest prime factor of i.
    # We iterate p from 2 to max_val; if lpf[p] is still 0, p is prime,
    # and we set lpf[j] = p for all multiples j of p.
    # Since we go in increasing p, the last assignment gives the largest prime factor.
    lpf = [0] * (max_val + 1)

    for p in range(2, max_val + 1):
        if lpf[p] == 0:  # p is prime
            for j in range(p, max_val + 1, p):
                lpf[j] = p

    # Scan for the minimum max(LPF(n), LPF(n+1)) over n in [10000, 1000000]
    best_n = 10000
    min_score = lpf[10000] if lpf[10000] > lpf[10001] else lpf[10001]

    for n in range(10001, limit + 1):
        a = lpf[n]
        b = lpf[n + 1]
        score = a if a > b else b
        if score < min_score:
            min_score = score
            best_n = n
            
            # If we find a pair with max LPF 5, it is the theoretical minimum 
            # for n >= 10000 (as consecutive 3-smooth numbers only occur at 8,9).
            # Thus, we can terminate early.
            if min_score == 5:
                break

        # Periodic time check to avoid exceeding deadline
        if (n - 10001) % 50000 == 0:
            if time.time() - start_time > deadline:
                break

    return best_n
