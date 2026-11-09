import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_MIN = 10000
    N_MAX = 1000000
    MAX_N = N_MAX + 1
    
    # Compute smallest prime factor for all numbers using standard sieve
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number by factorization
    powerful_part = [1] * (MAX_N + 1)
    
    for n in range(2, MAX_N + 1):
        x = n
        pp = 1
        while x > 1:
            p = spf[x]
            exp = 0
            while x % p == 0:
                x //= p
                exp += 1
            if exp >= 2:
                pp *= (p ** exp)
        powerful_part[n] = pp
    
    # Find best n
    best_n = N_MIN
    best_val = powerful_part[N_MIN] * powerful_part[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
