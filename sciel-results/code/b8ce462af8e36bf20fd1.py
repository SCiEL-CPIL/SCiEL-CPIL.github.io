import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin

    N = 1000000
    max_val = 1
    best_n = 10000

    pp = [1] * (N + 2)

    limit = int(math.isqrt(N)) + 1
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))

    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    for p in primes:
        p2 = p * p
        if p2 > N + 1:
            continue

        pe = p2
        while pe <= N + 1:
            pe_next = pe * p
            if pe_next > N + 1:
                for m in range(pe, N + 2, pe):
                    pp[m] *= pe
            else:
                # k not divisible by p: iterate in blocks
                max_k = (N + 1) // pe
                # For each block of p consecutive k values, skip the one divisible by p
                # k = j*p + r, r in 1..p-1
                for j in range(0, (max_k + p - 1) // p):
                    k_start = j * p + 1
                    k_end = min((j + 1) * p, max_k + 1)
                    for r in range(k_start, k_end):
                        pp[r * pe] *= pe
            pe = pe_next
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    max_prod = 0
    for n in range(10000, N + 1):
        prod = pp[n] * pp[n+1]
        if prod > max_prod:
            max_prod = prod
            best_n = n
        if time.time() > deadline:
            break

    return best_n
