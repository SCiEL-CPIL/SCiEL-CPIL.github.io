import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 110.0

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def compute_phi_sieve(n):
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        limit = int(np.sqrt(n)) + 1
        for i in range(2, limit):
            if is_prime[i]:
                is_prime[i * i::i] = False
        phi = np.arange(n + 1, dtype=np.int64)
        primes = np.where(is_prime)[0]
        for p in primes:
            if p > n:
                break
            multiples = np.arange(p, n + 1, p)
            phi[multiples] -= phi[multiples] // p
        return phi

    def find_longest_run(phi_arr, n):
        last_seen = np.full(n + 1, -1, dtype=np.int64)
        left = 0
        best_len = 1
        best_start = 1
        for i in range(1, n + 1):
            val = int(phi_arr[i])
            prev = last_seen[val]
            if prev >= left:
                left = prev + 1
            last_seen[val] = i
            run_len = i - left + 1
            if run_len > best_len:
                best_len = run_len
                best_start = left
        return best_len, best_start

    best_len = 1
    best_start = 1

    for n in [100000, 500000, 1000000, 5000000, 10000000, 20000000]:
        if time_remaining() < 50:
            break
        try:
            phi_arr = compute_phi_sieve(n)
        except Exception:
            break
        if time_remaining() < 20:
            break
        k, s = find_longest_run(phi_arr, n)
        if k > best_len:
            best_len = k
            best_start = s
        del phi_arr

    return {'start': best_start, 'length': best_len}
