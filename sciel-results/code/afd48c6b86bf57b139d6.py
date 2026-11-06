"""Structural redesign: randomized greedy with simulated-annealing-style perturbations.

The seed program is a single deterministic greedy pass that gets stuck in a local
optimum. We instead:
1. Build an initial set via a randomized greedy that tries candidates in random order.
2. Then run a local search / simulated annealing loop that tries to add a new element,
   or swap an existing element for a new one, to escape local optima.
3. Use a deadline guard to stay within the 150 s wall-clock limit.
"""

import time
import random

DEADLINE_SECONDS = 140.0  # safety margin below the 150 s kill


def _initial_sums(a):
    """Return the set of all pairwise sums a_i + a_j for i <= j."""
    s = set()
    n = len(a)
    for i in range(n):
        for j in range(i, n):
            s.add(a[i] + a[j])
    return s


def _try_add(a, sums, x):
    """Try to add x to set a. Return (success, new_a, new_sums) or (False, None, None)."""
    if x in a:
        return False, None, None
    # New sums introduced by x: x + y for all y in a, and 2*x
    new_sums = {x + y for y in a} | {2 * x}
    if new_sums & sums:
        return False, None, None
    return True, a + [x], sums | new_sums


def _randomized_greedy(seed_val):
    """Build a Sidon set containing {1,2,5,11} using randomized greedy."""
    rng = random.Random(seed_val)
    a = [1, 2, 5, 11]
    sums = _initial_sums(a)
    candidates = list(range(12, 285))
    rng.shuffle(candidates)
    for x in candidates:
        ok, new_a, new_sums = _try_add(a, sums, x)
        if ok:
            a = new_a
            sums = new_sums
    return a, sums


def _local_search(a, sums, rng, max_iters=200000):
    """Local search: try to add elements or perform swaps to increase size."""
    a = list(a)
    sums = set(sums)
    best_size = len(a)
    best_a = list(a)
    best_sums = set(sums)

    for _ in range(max_iters):
        # Strategy 1: try to add a random unused element
        used = set(a)
        unused = [x for x in range(1, 285) if x not in used]
        if unused:
            x = rng.choice(unused)
            ok, new_a, new_sums = _try_add(a, sums, x)
            if ok:
                a = new_a
                sums = new_sums
                if len(a) > best_size:
                    best_size = len(a)
                    best_a = list(a)
                    best_sums = set(sums)
                continue

        # Strategy 2: try a swap – remove one element, add another
        i = rng.randrange(len(a))
        removed = a[i]
        a_without = a[:i] + a[i+1:]
        sums_without = set()
        n2 = len(a_without)
        for p in range(n2):
            for q in range(p, n2):
                sums_without.add(a_without[p] + a_without[q])

        used = set(a_without)
        unused = [x for x in range(1, 285) if x not in used]
        if not unused:
            continue
        x = rng.choice(unused)
        ok, new_a, new_sums = _try_add(a_without, sums_without, x)
        if ok:
            a = new_a
            sums = new_sums
            if len(a) > best_size:
                best_size = len(a)
                best_a = list(a)
                best_sums = set(sums)

    return best_a, best_sums


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    best_a = [1, 2, 5, 11]
    best_size = 4

    # Try multiple randomized greedy starts, then local search on the best ones
    rng = random.Random(42)
    num_starts = 30

    for s in range(num_starts):
        if time.time() > deadline - 10:
            break

        a, sums = _randomized_greedy(s)
        if len(a) > best_size:
            best_a = list(a)
            best_size = len(a)

        # Run local search if time permits
        if time.time() > deadline - 15:
            break

        a2, sums2 = _local_search(a, sums, rng, max_iters=50000)
        if len(a2) > best_size:
            best_a = list(a2)
            best_size = len(a2)

    # Final validation
    a = best_a
    s = set()
    for i in range(len(a)):
        for j in range(i, len(a)):
            s.add(a[i] + a[j])
    if len(s) != len(a) * (len(a) + 1) // 2:
        # Fallback to seed result
        a = [1, 2, 5, 11]
        sums = _initial_sums(a)
        for x in range(12, 285):
            new = {x + y for y in a} | {2 * x}
            if len(new) == len(a) + 1 and not new & sums:
                a.append(x)
                sums |= new

    return a


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Set: {result}")
