import time
import random
import numpy as np

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, v in enumerate(sieve) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    primes = _primes_up_to(1000)
    n_primes = len(primes)

    WORK_LIMIT = 30000

    # Initialize with seed strategy
    best_res = np.array([((i + 2) // 2) % primes[i] for i in range(n_primes)], dtype=np.int64)

    # Build coverage array
    cov = np.zeros(WORK_LIMIT + 1, dtype=np.int16)
    for i in range(n_primes):
        p = primes[i]
        r = best_res[i]
        start = p if r == 0 else r
        if start <= WORK_LIMIT:
            cov[start::p] += 1

    def get_L(cov, limit):
        bad = np.where(cov[1:limit+1] < 2)[0]
        return limit if len(bad) == 0 else bad[0]

    best_L = get_L(cov, WORK_LIMIT)

    # Local search with simulated annealing
    no_improve = 0
    temp = 5.0
    min_temp = 0.05

    while time.time() < deadline:
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = int(best_res[i])
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue

        # Remove old coverage
        old_start = p if old_r == 0 else old_r
        if old_start <= WORK_LIMIT:
            cov[old_start::p] -= 1

        # Add new coverage
        new_start = p if new_r == 0 else new_r
        if new_start <= WORK_LIMIT:
            cov[new_start::p] += 1

        new_L = get_L(cov, WORK_LIMIT)

        if new_L >= best_L:
            best_res[i] = new_r
            best_L = new_L
            no_improve = 0
            temp = max(min_temp, temp * 0.999)
        else:
            delta = best_L - new_L
            if random.random() < np.exp(-delta / max(temp, 0.01)):
                best_res[i] = new_r
                no_improve += 1
            else:
                # Revert
                if new_start <= WORK_LIMIT:
                    cov[new_start::p] -= 1
                if old_start <= WORK_LIMIT:
                    cov[old_start::p] += 1
                no_improve += 1

    # Final full evaluation
    full_limit = 200000
    cov_full = np.zeros(full_limit + 1, dtype=np.int16)
    for i in range(n_primes):
        p = primes[i]
        r = int(best_res[i])
        start = p if r == 0 else r
        if start <= full_limit:
            cov_full[start::p] += 1

    final_L = get_L(cov_full, full_limit)

    return list(best_res)
