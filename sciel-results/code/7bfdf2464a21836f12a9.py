import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 20
    maximum = 10000

    def _score(A):
        """Return max(|A+A|, |A*A|) for list A of length n."""
        ss = set()
        ps = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                ss.add(ai + A[j])
                ps.add(ai * A[j])
        return max(len(ss), len(ps))

    def _score_early(A, bound):
        """Score with early exit if either set exceeds bound."""
        ss = set()
        ps = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                ss.add(ai + A[j])
                if len(ss) >= bound:
                    return bound
                ps.add(ai * A[j])
                if len(ps) >= bound:
                    return bound
        return max(len(ss), len(ps))

    def _valid(A):
        if len(A) != n:
            return False
        if len(set(A)) != n:
            return False
        if any(x < 1 or x > maximum for x in A):
            return False
        return True

    # Generate 5-smooth numbers (products of 2, 3, 5) up to maximum
    smooth = set()
    for a in range(1, maximum + 1):
        x = a
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.add(a)
    smooth_list = sorted(smooth)

    # Build candidate pool: smooth numbers + some structured values
    candidate_pool = set(smooth_list)
    # Add small integers and some multiples
    for k in range(1, 200):
        candidate_pool.add(k)
    for base in [2, 3, 4, 5, 6, 7]:
        for m in range(1, 100):
            v = base * m
            if v <= maximum:
                candidate_pool.add(v)
    # Add some random values for diversity
    rng_init = random.Random(999)
    for _ in range(500):
        candidate_pool.add(rng_init.randint(1, maximum))
    candidate_pool = sorted(candidate_pool)

    # Limit pool for speed
    if len(candidate_pool) > 3000:
        # Keep small values, smooth numbers, and some spread
        small = [x for x in candidate_pool if x <= 500]
        smooth_in_pool = [x for x in candidate_pool if x in smooth]
        large_sample = candidate_pool[::max(1, len(candidate_pool) // 500)]
        candidate_pool = sorted(set(small + smooth_in_pool + large_sample))

    # AP initialization: minimizes sumset to theoretical minimum of 3n-3 = 57
    # For n=20, AP gives sumset size 39 (2n-1 = 39)
    # Actually for AP {a, a+d, ..., a+(n-1)d}, sumset = {2a, 2a+d, ..., 2a+2(n-1)d} = 2n-1 = 39
    # Product set for AP is larger, but this gives us a strong starting point

    # Generate multiple AP-based candidates
    candidates = []
    for start in range(1, 50):
        for d in range(1, 10):
            A = [start + i * d for i in range(n)]
            if max(A) <= maximum:
                candidates.append(sorted(A))

    # Also try AP with larger gaps
    for start in range(1, 20):
        for d in [10, 15, 20, 25, 30, 40, 50]:
            A = [start + i * d for i in range(n)]
            if max(A) <= maximum:
                candidates.append(sorted(A))

    # Evaluate all AP candidates
    best_A = None
    best_score = float('inf')
    for A in candidates:
        if not _valid(A):
            continue
        sc = _score(A)
        if sc < best_score:
            best_score = sc
            best_A = A

    if best_A is None:
        best_A = list(range(1, n + 1))
        best_score = _score(best_A)

    # Simulated annealing local search
    rng = random.Random(42)
    current_A = list(best_A)
    current_score = best_score
    global_best_A = list(best_A)
    global_best_score = best_score

    # Annealing parameters
    T = 50.0  # initial temperature
    cooling_rate = 0.9995
    min_T = 0.01

    step = 0
    while time.time() < deadline and T > min_T:
        # Pick a random position to modify
        i = rng.randint(0, n - 1)
        old_val = current_A[i]

        # Choose new value from candidate pool (biased toward smooth numbers and nearby values)
        if rng.random() < 0.7:
            # Pick from smooth numbers near old_val or random smooth
            smooth_near = [x for x in smooth_list if abs(x - old_val) < max(100, old_val // 2)]
            if smooth_near:
                new_val = rng.choice(smooth_near)
            else:
                new_val = rng.choice(smooth_list)
        elif rng.random() < 0.5:
            # Nearby value
            delta = rng.randint(-20, 20)
            new_val = old_val + delta
            if new_val < 1 or new_val > maximum:
                new_val = rng.randint(1, maximum)
        else:
            # Random from pool
            new_val = rng.choice(candidate_pool)

        if new_val == old_val or new_val in current_A:
            continue

        # Try the swap
        current_A[i] = new_val
        if not _valid(current_A):
            current_A[i] = old_val
            continue

        new_score = _score_early(current_A, current_score)

        # Acceptance criterion
        if new_score < current_score:
            # Always accept improvement
            current_score = new_score
            if new_score < global_best_score:
                global_best_score = new_score
                global_best_A = list(current_A)
        else:
            # Accept worse with probability exp(-(new_score - current_score) / T)
            delta_score = new_score - current_score
            if delta_score > 0:
                prob = math.exp(-delta_score / T)
                if rng.random() < prob:
                    current_score = new_score
                else:
                    current_A[i] = old_val
                    continue

        T *= cooling_rate
        step += 1

    # Final local search: greedy improvement pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline - 2:
                break
            old_val = current_A[i]
            # Try all values in a focused subset of the pool
            # Focus on smooth numbers and nearby values
            tried = set()
            for nv in candidate_pool:
                if time.time() > deadline - 1:
                    break
                if nv == old_val or nv in current_A:
                    continue
                if nv in tried:
                    continue
                tried.add(nv)
                current_A[i] = nv
                if not _valid(current_A):
                    current_A[i] = old_val
                    continue
                sc = _score_early(current_A, current_score)
                if sc < current_score:
                    current_score = sc
                    improved = True
                    if sc < global_best_score:
                        global_best_score = sc
                        global_best_A = list(current_A)
                    break
                current_A[i] = old_val
            if improved:
                break

    # Ensure we return the best found
    if global_best_score < current_score:
        return sorted(global_best_A)
    else:
        return sorted(current_A)
