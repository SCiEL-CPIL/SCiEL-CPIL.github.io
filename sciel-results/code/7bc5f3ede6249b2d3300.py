import time
import random
from itertools import combinations

def _is_sidon_fast(arr):
    n = len(arr)
    seen = set()
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            s = ai + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(arr):
    n = len(arr)
    for k in range(n, 0, -1):
        for subset in combinations(arr, k):
            if _is_sidon_fast(subset):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    best_set = list(range(12))
    best_size = _max_sidon_size(best_set)

    random.seed(777)

    candidates = []

    # Small-range sets
    for _ in range(300):
        span = random.randint(12, 70)
        lo = random.randint(0, max(0, 128 - span))
        hi = min(lo + span, 128)
        if hi - lo + 1 >= 12:
            candidates.append(sorted(random.sample(range(lo, hi + 1), 12)))

    # APs
    for step in range(2, 35):
        max_start = 128 - 11 * step
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, max(1, step // 2)):
            arr = [start + i * step for i in range(12)]
            if max(arr) <= 128:
                candidates.append(arr)

    # Random full-range
    for _ in range(300):
        candidates.append(sorted(random.sample(range(129), 12)))

    # Clustered: two groups
    for _ in range(200):
        lo1 = random.randint(0, 40)
        hi1 = lo1 + random.randint(8, 20)
        lo2 = random.randint(70, 100)
        hi2 = lo2 + random.randint(8, 20)
        if hi1 < 129 and hi2 < 129:
            half = 6
            g1 = random.sample(range(lo1, min(hi1, 129) + 1), min(half, hi1 - lo1 + 1))
            g2 = random.sample(range(lo2, min(hi2, 129) + 1), min(half, hi2 - lo2 + 1))
            if len(g1) + len(g2) >= 12:
                while len(g1) < half and lo1 < hi1:
                    g1.append(random.randint(lo1, hi1))
                while len(g2) < half and lo2 < hi2:
                    g2.append(random.randint(lo2, hi2))
                arr = sorted(g1[:half] + g2[:half])
                if len(set(arr)) == 12:
                    candidates.append(arr)

    # Deduplicate
    seen_sets = set()
    unique_candidates = []
    for c in candidates:
        t = tuple(c)
        if t not in seen_sets:
            seen_sets.add(t)
            unique_candidates.append(c)

    # Evaluate candidates
    for arr in unique_candidates:
        if time.time() > deadline - 20.0:
            break
        size = _max_sidon_size(arr)
        if size < best_size:
            best_size = size
            best_set = arr[:]
            if best_size <= 3:
                break

    # Local search from best
    if best_set and time.time() < deadline - 10.0:
        current = best_set[:]
        current_size = best_size
        stall = 0
        max_stall = 100

        while stall < max_stall and time.time() < deadline - 5.0:
            stall += 1
            improved = False
            n = len(current)

            # Single replacement
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                old = current[i]
                for _ in range(30):
                    new_val = random.randint(0, 128)
                    if new_val == old or new_val in current:
                        continue
                    current[i] = new_val
                    current.sort()
                    ns = _max_sidon_size(current)
                    if ns < current_size:
                        current_size = ns
                        improved = True
                        if current_size < best_size:
                            best_size = current_size
                            best_set = current[:]
                            stall = 0
                            if best_size <= 2:
                                return best_set
                    else:
                        current[i] = old
                        current.sort()

            # Pairwise swap
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(i + 1, n):
                        current[i], current[j] = current[j], current[i]
                        current.sort()
                        ns = _max_sidon_size(current)
                        if ns < current_size:
                            current_size = ns
                            improved = True
                            if current_size < best_size:
                                best_size = current_size
                                best_set = current[:]
                                stall = 0
                                if best_size <= 2:
                                    return best_set
                        else:
                            current[i], current[j] = current[j], current[i]
                            current.sort()

        if current_size < best_size:
            best_size = current_size
            best_set = current[:]

    return best_set
