import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    MAX_VAL = 10000
    N = 12
    MAX_SUM = 20000

    # SPF sieve
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def prime_factors(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def full_score(arr):
        all_primes = set()
        for i in range(N):
            for j in range(i+1, N):
                all_primes |= prime_factors(arr[i] + arr[j])
        return all_primes

    # Seeds
    seeds = [
        list(range(1, N+1)),
        [6*(i+1) for i in range(N)],
        [12*(i+1) for i in range(N)],
        [2**(i+1) for i in range(N)],
        [30*(i+1) for i in range(N)],
        sorted(random.Random(42).sample(range(1, MAX_VAL+1), N)),
        [2*(i+1) for i in range(N)],
        [10*(i+1) for i in range(N)],
    ]

    best_arr = None
    best_score = float('inf')

    for seed in seeds:
        if time.time() - start_time > DEADLINE:
            break
        s = len(full_score(seed))
        if s < best_score:
            best_score = s
            best_arr = seed[:]

    if best_arr is None:
        best_arr = list(range(1, N+1))
        best_score = len(full_score(best_arr))

    # Local search with incremental updates
    current = best_arr[:]
    current_score = best_score

    # Build sums dict: key=(i,j) with i<j, value=set of primes
    sums_dict = {}
    for i in range(N):
        for j in range(i+1, N):
            sums_dict[(i,j)] = prime_factors(current[i] + current[j])

    # prime_count: p -> number of sums that have p
    prime_count = defaultdict(int)
    for pf in sums_dict.values():
        for p in pf:
            prime_count[p] += 1

    def distinct_count(pc):
        return sum(1 for c in pc.values() if c > 0)

    cur_distinct = distinct_count(prime_count)

    no_improve = 0
    max_no_improve = 2000
    rng = random.Random(12345)

    while time.time() - start_time < DEADLINE:
        # Pick index to replace
        idx = rng.randint(0, N-1)

        # Generate candidate
        r = rng.random()
        if r < 0.6:
            candidate = rng.randint(1, MAX_VAL)
        elif r < 0.85:
            base = current[rng.randint(0, N-1)]
            delta = rng.randint(-100, 100)
            candidate = max(1, min(MAX_VAL, base + delta))
        else:
            # Try to be a multiple of small numbers
            base_mult = rng.choice([2, 3, 4, 5, 6, 10, 12, 30])
            candidate = base_mult * rng.randint(1, MAX_VAL // base_mult)

        if candidate in current:
            continue

        # Remove old sums involving idx
        old_sums = []
        for j in range(N):
            if j == idx:
                continue
            key = (min(idx,j), max(idx,j))
            pf = sums_dict[key]
            old_sums.append((key, pf))
            for p in pf:
                prime_count[p] -= 1
                if prime_count[p] == 0:
                    del prime_count[p]

        # Add new sums
        new_sums = []
        for j in range(N):
            if j == idx:
                continue
            s_val = candidate + current[j]
            pf = prime_factors(s_val)
            key = (min(idx,j), max(idx,j))
            new_sums.append((key, pf))
            for p in pf:
                if p not in prime_count:
                    prime_count[p] = 0
                prime_count[p] += 1

        new_distinct = distinct_count(prime_count)

        if new_distinct < cur_distinct:
            # Accept
            for key, pf in old_sums:
                del sums_dict[key]
            for key, pf in new_sums:
                sums_dict[key] = pf
            current[idx] = candidate
            cur_distinct = new_distinct
            no_improve = 0

            if cur_distinct < best_score:
                best_score = cur_distinct
                best_arr = current[:]
        else:
            # Reject: undo
            for key, pf in new_sums:
                del sums_dict[key]
                for p in pf:
                    prime_count[p] -= 1
                    if prime_count[p] == 0:
                        del prime_count[p]
            for key, pf in old_sums:
                sums_dict[key] = pf
                for p in pf:
                    if p not in prime_count:
                        prime_count[p] = 0
                    prime_count[p] += 1

            no_improve += 1
            if no_improve > max_no_improve:
                if rng.random() < 0.4:
                    # Restart from best
                    current = best_arr[:]
                    cur_distinct = best_score
                    sums_dict = {}
                    for i in range(N):
                        for j in range(i+1, N):
                            sums_dict[(i,j)] = prime_factors(current[i] + current[j])
                    prime_count = defaultdict(int)
                    for pf in sums_dict.values():
                        for p in pf:
                            prime_count[p] += 1
                    no_improve = 0
                else:
                    max_no_improve = max(200, max_no_improve // 2)

    return best_arr
