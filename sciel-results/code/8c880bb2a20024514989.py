import time
from math import isqrt
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def generate_powerful(limit):
        """Generate all powerful numbers up to limit."""
        powerful = set()
        powerful.add(1)

        b_max = int(limit ** (1/3)) + 2

        # Generate square-free numbers up to b_max
        is_sf = [True] * (b_max + 1)
        is_sf[0] = False
        if b_max >= 1:
            is_sf[1] = True

        for p in range(2, isqrt(b_max) + 1):
            if is_sf[p]:
                p2 = p * p
                for j in range(p2, b_max + 1, p2):
                    is_sf[j] = False

        sf_list = [i for i in range(2, b_max + 1) if is_sf[i]]

        for b in sf_list:
            b3 = b ** 3
            if b3 > limit:
                continue
            a_max = isqrt(limit // b3)
            for a in range(1, a_max + 1):
                val = a * a * b3
                if val <= limit:
                    powerful.add(val)

        powerful.add(1)
        return sorted(powerful)

    best_N = 2
    best_count = 1
    best_limit_used = 50_000_000

    limits_to_try = [50_000_000, 100_000_000, 150_000_000, 200_000_000]

    for limit in limits_to_try:
        if time_left() < 25:
            break

        powerful = generate_powerful(limit)
        K = len(powerful)
        print(f"Generated {K} powerful numbers up to {limit}")

        if time_left() < 20:
            break

        arr = np.array(powerful, dtype=np.int64)

        sum_counts = defaultdict(int)

        t0 = time.time()
        for i in range(K):
            a = arr[i]
            remaining = arr[i:]
            sums = a + remaining
            unique_sums, counts = np.unique(sums, return_counts=True)
            for s, c in zip(unique_sums, counts):
                sum_counts[s] += c

        elapsed = time.time() - t0
        max_count = max(sum_counts.values()) if sum_counts else 0
        print(f"Max count: {max_count}, counting took {elapsed:.2f}s, time_left={time_left():.1f}s")

        if max_count > best_count:
            for s, c in sum_counts.items():
                if c == max_count:
                    best_N = s
                    best_count = max_count
                    best_limit_used = limit
                    break

    # Reconstruct pairs for best_N
    if time_left() > 10:
        powerful_full = generate_powerful(best_N)
    else:
        powerful_full = generate_powerful(best_limit_used)

    powerful_set = set(powerful_full)
    pairs = []
    for a in powerful_full:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])

    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': pairs}
