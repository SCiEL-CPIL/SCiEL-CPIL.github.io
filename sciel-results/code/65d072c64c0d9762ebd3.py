"""Optimized search for integer sum-product extremal sets."""
import time
import random
import math

# Deadline guard: evaluator kills after 90s wall-clock. We aim to return well before.
_DEADLINE_SECONDS = 85.0


def _objective(a):
    """Compute max(|A+A|, |A*A|) for set a (list of distinct positive ints)."""
    n = len(a)
    sums = set()
    prods = set()
    for i in range(n):
        ai = a[i]
        for j in range(i, n):
            aj = a[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))


def _is_valid(a, maximum=10000):
    """Check that a has 20 distinct positive integers <= maximum."""
    if len(a) != 20:
        return False
    if len(set(a)) != 20:
        return False
    for x in a:
        if x < 1 or x > maximum:
            return False
    return True


def _generate_structured_candidates():
    """
    Generate candidate sets from structured families:
    - Short arithmetic progressions combined with geometric progressions
    - Powers of small bases
    - Mixed structured sets designed to create collisions in A+A and A*A
    """
    candidates = []

    # Family 1: Union of two arithmetic progressions
    # AP1: start1, start1+step1, ..., start1+(len1-1)*step1
    # AP2: start2, start2+step2, ..., start2+(len2-1)*step2
    for len1 in range(5, 16):
        len2 = 20 - len1
        if len2 < 1:
            continue
        for step1 in [1, 2, 3, 4, 5, 6, 8, 10]:
            for step2 in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]:
                for start1 in [1, 2, 3, 5, 7, 10]:
                    for start2 in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
                        ap1 = [start1 + i * step1 for i in range(len1)]
                        ap2 = [start2 + i * step2 for i in range(len2)]
                        combined = sorted(set(ap1 + ap2))
                        if len(combined) == 20 and all(1 <= x <= 10000 for x in combined):
                            candidates.append(combined)

    # Family 2: Powers of small bases (geometric-like)
    for base in [2, 3, 4, 5]:
        powers = [base ** i for i in range(20)]
        if all(x <= 10000 for x in powers):
            candidates.append(powers)
        else:
            # Take first k powers that fit, then add some structured extras
            k = 0
            while k < 20 and base ** k <= 10000:
                k += 1
            if k >= 5:
                base_powers = [base ** i for i in range(k)]
                # Add some multiples or offsets to fill to 20
                remaining = 20 - k
                extra = []
                for m in range(1, 100):
                    val = base_powers[-1] + m
                    if val <= 10000 and val not in base_powers and val not in extra:
                        extra.append(val)
                        if len(extra) >= remaining:
                            break
                combined = sorted(base_powers + extra)
                if len(combined) == 20 and len(set(combined)) == 20:
                    candidates.append(combined)

    # Family 3: Small numbers with multiplicative structure
    # Use numbers with many small prime factors to create product collisions
    for base in [2, 3, 6, 12, 24, 48, 96]:
        structured = []
        for i in range(1, 50):
            val = base * i
            if val <= 10000 and val not in structured:
                structured.append(val)
            if len(structured) >= 20:
                break
        if len(structured) == 20:
            candidates.append(sorted(structured))

    # Family 4: Mix of small integers and powers
    small = list(range(1, 11))  # 1-10
    for base in [2, 3, 4]:
        powers = [base ** i for i in range(1, 11) if base ** i <= 10000]
        combined = sorted(set(small + powers))
        if len(combined) == 20:
            candidates.append(combined)
        elif len(combined) > 10:
            # Trim to 20
            if len(combined) >= 20:
                candidates.append(sorted(combined[:20]))

    # Family 5: Arithmetic progression with a twist (non-uniform steps)
    for start in [1, 2, 3]:
        for step in [1, 2, 3]:
            ap = [start + i * step for i in range(15)]
            if all(x <= 10000 for x in ap):
                # Add 5 more numbers that create sum collisions
                last = ap[-1]
                extras = [last + i * step for i in range(1, 6)]
                combined = sorted(set(ap + extras))
                if len(combined) == 20 and all(x <= 10000 for x in combined):
                    candidates.append(combined)

    # Family 6: Dense small set
    for end in [20, 25, 30, 40, 50]:
        candidates.append(list(range(1, end + 1))[:20])

    return candidates


def _local_search(a, time_limit, rng):
    """
    Local search with random replacements and nudges.
    Returns the best set found within time_limit.
    """
    best = list(a)
    best_obj = _objective(best)
    start_time = time.time()

    while time.time() - start_time < time_limit:
        # Generate a neighbor
        move_type = rng.random()

        if move_type < 0.5:
            # Replace one element with a random valid number
            idx = rng.randrange(20)
            old_val = best[idx]
            new_val = rng.randint(1, 10000)
            if new_val == old_val:
                continue
            # Check if new_val is already in the set
            if new_val in best:
                continue
            neighbor = list(best)
            neighbor[idx] = new_val
        elif move_type < 0.8:
            # Nudge one element by a small amount
            idx = rng.randrange(20)
            delta = rng.choice([-3, -2, -1, 1, 2, 3, 5, 10])
            new_val = best[idx] + delta
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in best:
                continue
            neighbor = list(best)
            neighbor[idx] = new_val
        else:
            # Swap two elements with random values
            idx1, idx2 = rng.sample(range(20), 2)
            v1 = rng.randint(1, 10000)
            v2 = rng.randint(1, 10000)
            if v1 in best or v2 in best or v1 == v2:
                continue
            neighbor = list(best)
            neighbor[idx1] = v1
            neighbor[idx2] = v2

        if not _is_valid(neighbor):
            continue

        obj = _objective(neighbor)
        if obj < best_obj:
            best = neighbor
            best_obj = obj

    return best, best_obj


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS

    rng = random.Random(42)

    # Generate structured candidates
    candidates = _generate_structured_candidates()

    # Also add some random candidates
    for _ in range(50):
        cand = sorted(rng.sample(range(1, 10001), 20))
        candidates.append(cand)

    # Evaluate all candidates
    best = None
    best_obj = float('inf')

    for cand in candidates:
        if time.time() > deadline - 5:
            break
        if not _is_valid(cand):
            continue
        obj = _objective(cand)
        if obj < best_obj:
            best_obj = obj
            best = list(cand)

    if best is None:
        # Fallback
        best = list(range(1, 21))
        best_obj = _objective(best)

    # Local search from the best candidate
    remaining_time = deadline - time.time() - 2  # 2s safety margin
    if remaining_time > 1:
        best, best_obj = _local_search(best, remaining_time, rng)

    # Additional local search from a few other good candidates if time permits
    remaining_time = deadline - time.time() - 2
    if remaining_time > 3:
        # Try local search from the 2nd and 3rd best candidates
        scored = []
        for cand in candidates[:20]:
            if not _is_valid(cand):
                continue
            obj = _objective(cand)
            scored.append((obj, cand))
        scored.sort()
        for obj, cand in scored[1:3]:
            if time.time() > deadline - 3:
                break
            remaining = deadline - time.time() - 2
            if remaining < 1:
                break
            local_best, local_obj = _local_search(cand, remaining, rng)
            if local_obj < best_obj:
                best_obj = local_obj
                best = local_best

    return best
