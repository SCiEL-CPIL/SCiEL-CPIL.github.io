"""Rewritten program: structural local search with arithmetic-progression seeding."""
import time
import random
import numpy as np


def _set_metrics(A):
    """Return (sumset_size, productset_size, max) for a list of ints."""
    arr = np.array(A, dtype=np.int64)
    # Sumset: all a_i + a_j for i <= j
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods), max(len(sums), len(prods))


def _seed_set(seed=42):
    """
    Generate an initial set of 20 distinct positive integers <= 10000.
    Uses a mix of arithmetic progressions with different step sizes
    to encourage additive collisions while keeping multiplicative spread.
    """
    rng = random.Random(seed)
    # Strategy: combine short APs
    # AP1: small numbers, step 1 -> dense sums
    # AP2: medium, larger step -> different sum structure
    # AP3: larger, very large step -> controlled products
    candidates = []

    # Block 1: 8 consecutive small numbers (step 1)
    base1 = rng.randint(1, 30)
    for i in range(8):
        candidates.append(base1 + i)

    # Block 2: 6 numbers, step 3
    base2 = rng.randint(40, 80)
    for i in range(6):
        candidates.append(base2 + 3 * i)

    # Block 3: 4 numbers, step 7
    base3 = rng.randint(100, 200)
    for i in range(4):
        candidates.append(base3 + 7 * i)

    # Block 4: 2 numbers, step 13
    base4 = rng.randint(300, 600)
    for i in range(2):
        candidates.append(base4 + 13 * i)

    # Deduplicate and ensure all <= 10000 and positive
    seen = set()
    result = []
    for c in candidates:
        if 1 <= c <= 10000 and c not in seen:
            seen.add(c)
            result.append(c)

    # If we don't have 20, fill in
    if len(result) < 20:
        # Add more with random small steps
        while len(result) < 20:
            c = rng.randint(1, 10000)
            if c not in seen:
                seen.add(c)
                result.append(c)

    result = sorted(result[:20])
    return result


def _local_search(A, deadline, seed=0):
    """
    Local search: try swapping each element with a candidate from a
    pre-generated pool, accepting if it improves or ties on max(|A+A|,|A*A|).
    """
    rng = random.Random(seed)
    A = list(A)
    n = len(A)
    _, _, best_val = _set_metrics(A)

    # Build a candidate pool: numbers near the current set values
    pool = set()
    for a in A:
        for delta in range(-50, 51):
            c = a + delta
            if 1 <= c <= 10000:
                pool.add(c)
    # Also add some random candidates
    for _ in range(2000):
        pool.add(rng.randint(1, 10000))

    pool_list = sorted(pool - set(A))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle candidates for diversity
        rng.shuffle(pool_list)
        for idx in range(n):
            old_val = A[idx]
            for new_val in pool_list:
                if new_val == old_val or new_val in A:
                    continue
                # Try the swap
                A[idx] = new_val
                _, _, val = _set_metrics(A)
                if val < best_val:
                    best_val = val
                    improved = True
                    # Update pool: remove old_val if it's in pool, add new_val's neighbors
                    pool_list = [x for x in pool_list if x != new_val]
                    for delta in range(-20, 21):
                        c = new_val + delta
                        if 1 <= c <= 10000 and c != new_val and c not in A:
                            if c not in pool_list:
                                pool_list.append(c)
                    break
                else:
                    A[idx] = old_val

    return A, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit, safe margin

    best_set = None
    best_val = float('inf')

    # Try several seeds for different initial structures
    for seed in range(20):
        if time.time() >= deadline:
            break
        A = _seed_set(seed)
        A, val = _local_search(A, deadline, seed=seed + 1000)
        if val < best_val:
            best_val = val
            best_set = list(A)

    if best_set is None:
        best_set = _seed_set(0)

    return best_set
