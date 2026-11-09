import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    N_MAX = 1000000
    N_MIN = 10000
    max_val = N_MAX + 1  # We need powerful_part up to N_MAX+1

    # Linear sieve to find smallest prime factor (spf) for each number
    spf = list(range(max_val + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Compute powerful part for each number
    # powerful_part[n] = product of p^e where e is the exponent of prime p in n's factorization, for all p with e >= 2
    powerful_part = [1] * (max_val + 1)

    # For each number, factorize using spf and compute powerful part
    for n in range(2, max_val + 1):
        m = n
        pp = 1
        while m > 1:
            p = spf[m]
            exp = 0
            while m % p == 0:
                m //= p
                exp += 1
            if exp >= 2:
                pp *= (p ** exp)
        powerful_part[n] = pp

    # Scan for the best n in [N_MIN, N_MAX]
    best_val = 0
    best_n = N_MIN

    for n in range(N_MIN, N_MAX + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n
