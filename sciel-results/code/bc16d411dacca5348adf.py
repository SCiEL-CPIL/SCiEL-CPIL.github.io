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

def _count_collisions(arr):
    """Count the number of duplicate pair sums (higher = more collisions = harder to find Sidon)."""
    n = len(arr)
    sums = []
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            sums.append(ai + arr[j])
    # Count how many sums appear more than once
    from collections import Counter
    c = Counter(sums)
    total = 0
    for v in c.values():
        if v > 1:
            total += v * (v - 1) // 2
    return total

def _collision_score(arr):
    """Score for collision maximization: more collisions is better (lower max Sidon)."""
    return -_count_collisions(arr)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    best_set = list(range(12))
    best_size = _max_sidon_size(best_set)

    random.seed(42)

    candidates = []

    # Strategy 1: 6 pairs summing to same value (maximum collision structure)
    for S in range(130, 256):
        pairs = []
        used = set()
        for a in range(0, 129):
            b = S - a
            if b > a and 0 <= b <= 128 and a not in used and b not in used:
                pairs.append((a, b))
                used.add(a)
                used.add(b)
                if len(pairs) == 6:
                    break
        if len(pairs) == 6:
            arr = sorted([x for p in pairs for x in p])
            candidates.append(arr)

    # Strategy 2: Mixed collision structures - 3 groups of 4
    for _ in range(300):
        S1 = random.randint(130, 250)
        S2 = random.randint(130, 250)
        S3 = random.randint(130, 250)
        all_vals = []
        valid = True
        for S in [S1, S2, S3]:
            pairs = []
            used = set()
            for a in range(0, 129):
                b = S - a
                if b > a and 0 <= b <= 128 and a not in used and b not in used:
                    pairs.append((a, b))
                    used.add(a)
                    used.add(b)
                    if len(pairs) == 2:
                        break
            if len(pairs) != 2:
                valid = False
                break
            all_vals.extend([x for p in pairs for x in p])
        if valid and len(set(all_vals)) == 12:
            candidates.append(sorted(all_vals))

    # Strategy 3: Two dense clusters
    for _ in range(500):
        span1 = random.randint(12, 25)
        span2 = random.randint(12, 25)
        lo1 = random.randint(0, max(0, 50 - span1))
        lo2 = random.randint(78, max(78, 128 - span2))
        hi1 = min(lo1 + span1, 128)
        hi2 = min(lo2 + span2, 128)
        if hi1 - lo1 + 1 >= 6 and hi2 - lo2 + 1 >= 6:
            c1 = random.sample(range(lo1, hi1 + 1), 6)
            c2 = random.sample(range(lo2, hi2 + 1), 6)
            all_vals = c1 + c2
            if len(set(all_vals)) == 12:
                candidates.append(sorted(all_vals))

    # Strategy 4: Arithmetic progressions with small steps
    for step in range(2, 20):
        max_start = 128 - 11 * step
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, max(1, step // 3)):
            arr = [start + i * step for i in range(12)]
            if max(arr) <= 128:
                candidates.append(arr)

    # Strategy 5: Random clustered sets
    for _ in range(1000):
        span = random.randint(12, 40)
        lo = random.randint(0, max(0, 128 - span))
        hi = min(lo + span, 128)
        if hi - lo + 1 >= 12:
            candidates.append(sorted(random.sample(range(lo, hi + 1), 12)))

    # Strategy 6: Probabilistic collision maximization - start from random, greedily swap
    for _ in range(200):
        arr = sorted(random.sample(range(0, 129), 12))
        # Greedy local search to maximize collisions
        for _iter in range(30):
            if time.time() > deadline - 30.0:
                break
            best_swap = None
            best_score = _count_collisions(arr)
            for i in range(12):
                for new_val in random.sample([x for x in range(129) if x not in arr], min(20, 117)):
                    arr[i] = new_val
                    arr.sort()
                    score = _count_collisions(arr)
                    if score > best_score:
                        best_score = score
                        best_swap = (i, new_val)
                    arr[i] = arr[0]  # will be reset below
                    # Actually need to restore properly
                    break
                # Restore
                arr.sort()
            if best_swap is None:
                break
            i, new_val = best_swap
            arr[i] = new_val
            arr.sort()
        candidates.append(arr)

    # Deduplicate
    seen_sets = set()
    unique_candidates = []
    for c in candidates:
        t = tuple(c)
        if t not in seen_sets:
            seen_sets.add(t)
            unique_candidates.append(c)

    # Sort candidates by collision count (highest first) for better search
    unique_candidates.sort(key=lambda x: -_count_collisions(x))

    # Evaluate candidates
    for arr in unique_candidates:
        if time.time() > deadline - 15.0:
            break
        size = _max_sidon_size(arr)
        if size < best_size:
            best_size = size
            best_set = arr[:]
            if best_size <= 3:
                break

    # Local search: single replacement and pairwise swap
    if best_set and time.time() < deadline - 10.0:
        current = best_set[:]
        current_size = best_size
        stall = 0
        max_stall = 50

        while stall < max_stall and time.time() < deadline - 5.0:
            stall += 1
            improved = False
            n = len(current)

            # Single replacement
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                old = current[i]
                for _ in range(20):
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
