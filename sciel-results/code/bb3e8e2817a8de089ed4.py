import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    N = 1000000
    MAX_VAL = N + 24  # largest number whose smooth part we need

    # Smallest prime factor (SPF) sieve
    spf = [0] * (MAX_VAL + 1)
    for i in range(2, MAX_VAL + 1):
        if spf[i] == 0:
            for j in range(i, MAX_VAL + 1, i):
                if spf[j] == 0:
                    spf[j] = i

    # Smooth part: product of p^e for each prime p < 24 dividing x
    # (primes < 24: 2,3,5,7,11,13,17,19,23)
    smooth = [1] * (MAX_VAL + 1)
    for x in range(2, MAX_VAL + 1):
        v = x
        s = 1
        while v > 1:
            p = spf[v]
            if p < 24:
                while v % p == 0:
                    s *= p
                    v //= p
            else:
                break  # remaining factors are all >= 24, stop
        smooth[x] = s

    # Sliding window of size 24 over smooth[n+1 .. n+24]
    # n ranges over [1000, N]; window covers smooth[n+1..n+24]
    W = 24
    freq = {}
    # Initialize window for n = 1000: indices 1001..1024
    lo = 1000 + 1
    hi = lo + W - 1
    for x in range(lo, hi + 1):
        freq[smooth[x]] = freq.get(smooth[x], 0) + 1
    distinct = len(freq)
    best_distinct = distinct
    best_n = 1000

    n = 1000
    while n <= N:
        if time.time() > deadline:
            break
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            if best_distinct == 1:
                break  # can't do better than 1
        # Slide: remove smooth[n+1], add smooth[n+25]
        out_val = smooth[n + 1]
        in_val = smooth[n + 25]
        c = freq[out_val] - 1
        if c == 0:
            del freq[out_val]
        else:
            freq[out_val] = c
        freq[in_val] = freq.get(in_val, 0) + 1
        distinct = len(freq)
        n += 1

    return best_n
