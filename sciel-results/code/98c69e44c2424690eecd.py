"""Rewritten program: optimize sumset/difference-set exponent lower bound."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_sorted = sorted(set(U))
    n = len(U_sorted)
    if n == 0:
        return 0.0
    max_u = U_sorted[-1]
    if max_u == 0:
        return 1.0
    if n == 1:
        # U = {0}: sumset={0}, diffset={0}, ratio=1
        return 1.0

    # Compute sumset and difference set
    sumset = set()
    diffset = set()
    for i, a in enumerate(U_sorted):
        for b in U_sorted[i:]:  # sumset: a+b is symmetric
            sumset.add(a + b)
    for i, a in enumerate(U_sorted):
        for b in U_sorted:
            diffset.add(a - b)

    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_fast(U):
    """Faster score using numpy for larger sets."""
    U_arr = sorted(set(U))
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = U_arr[-1]
    if max_u == 0 or n == 1:
        return 1.0

    # Use numpy for vectorized computation
    import numpy as np
    arr = np.array(U_arr)
    sums = arr[:, None] + arr[None, :]
    diffs = arr[:, None] - arr[None, :]
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _local_search(U, max_iters=300, deadline=None, move_radius=30):
    """Improve set U via local search with element moves and insertions."""
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    max_u = max(U) if U else 0
    upper_bound = max_u + move_radius

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False

        # Try moving each non-zero element
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            # Try nearby values
            for delta in range(1, move_radius + 1):
                for new_val in (old_val - delta, old_val + delta):
                    if new_val <= 0 or new_val > upper_bound:
                        continue
                    if new_val in U:
                        continue
                    new_U = U[:]
                    new_U[i] = new_val
                    new_U = sorted(new_U)
                    s = _score_set_fast(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U
                        upper_bound = max(upper_bound, new_val + move_radius)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                U = best_U[:]
                break

        # Try inserting a new element
        if not improved:
            for new_val in range(1, upper_bound + 1):
                if new_val in U:
                    continue
                new_U = U + [new_val]
                new_U = sorted(new_U)
                s = _score_set_fast(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    upper_bound = max(upper_bound, new_val + move_radius)
                    improved = True
                    break
            if improved:
                U = best_U[:]

        if not improved:
            # Try removing an element (except 0)
            removed = False
            for i in range(1, len(U)):
                new_U = U[:i] + U[i+1:]
                s = _score_set_fast(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    U = new_U[:]
                    upper_bound = max(U) + move_radius
                    removed = True
                    break
            if not removed:
                break

    return sorted(set(best_U)), best_score

def _simulated_annealing(U, max_iters=500, deadline=None):
    """Simulated annealing for global search."""
    U = sorted(set(U))
    best_U = U[:]
    best_score = _score_set_fast(U)
    current_U = U[:]
    current_score = best_score
    T = 0.1
    cooling = 0.995
    upper_bound = max(U) + 40

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break

        # Generate a random move
        move_type = random.random()
        if move_type < 0.5:
            # Move an element
            idx = random.randint(1, len(current_U) - 1)
            new_val = random.randint(1, upper_bound)
            if new_val in current_U:
                continue
            new_U = current_U[:]
            new_U[idx] = new_val
            new_U = sorted(set(new_U))
        elif move_type < 0.75:
            # Insert element
            new_val = random.randint(1, upper_bound)
            if new_val in current_U:
                continue
            new_U = current_U + [new_val]
            new_U = sorted(new_U)
        else:
            # Remove element
            if len(current_U) <= 2:
                continue
            idx = random.randint(1, len(current_U) - 1)
            new_U = current_U[:idx] + current_U[idx+1:]
            new_U = sorted(new_U)

        s = _score_set_fast(new_U)
        delta_s = s - current_score

        if delta_s > 0 or random.random() < math.exp(delta_s / T):
            current_U = new_U
            current_score = s
            upper_bound = max(upper_bound, max(new_U) + 10)
            if s > best_score:
                best_score = s
                best_U = new_U[:]

        T *= cooling
        if T < 0.001:
            T = 0.001

    return sorted(set(best_U)), best_score

def _construct_golomb_inspired():
    """Construct sets inspired by Golomb rulers (all differences distinct)."""
    candidates = []

    # Known optimal Golomb rulers
    golomb_rulers = [
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 4, 10, 12, 17, 22, 29],
        [0, 1, 4, 10, 12, 17, 22, 29, 36],
    ]
    candidates.extend(golomb_rulers)

    # Generate more Golomb-like sets via greedy construction
    for n in range(3, 12):
        for seed in range(5):
            random.seed(seed * 100 + n)
            U = [0]
            last = 0
            for _ in range(n - 1):
                # Choose next element to avoid duplicate differences
                while True:
                    candidate = last + random.randint(1, 15)
                    # Check all differences are distinct
                    diffs = set()
                    valid = True
                    for i in range(len(U)):
                        for j in range(i + 1, len(U) + 1):
                            val = (U[i] if i < len(U) else candidate) - (U[j] if j < len(U) else 0)
                            if val in diffs:
                                valid = False
                                break
                            diffs.add(val)
                        if not valid:
                            break
                    if valid:
                        break
                    last += 1
                U.append(candidate)
                last = candidate
            if len(set(U)) == len(U):
                candidates.append(sorted(U))

    return candidates

def _construct_arithmetic_perturbations():
    """Construct sets that are perturbations of arithmetic progressions."""
    candidates = []

    for n in range(3, 15):
        for step in range(1, 6):
            base = [i * step for i in range(n)]
            # Apply random perturbations
            for trial in range(20):
                random.seed(trial * 1000 + n * 100 + step)
                U = base[:]
                for _ in range(random.randint(1, min(3, n - 1))):
                    idx = random.randint(1, n - 1)
                    U[idx] += random.choice([-2, -1, 1, 2])
                    if U[idx] <= 0:
                        U[idx] = 1
                U = sorted(set(U))
                if 0 in U and len(U) >= 3:
                    candidates.append(U)

    return candidates

def _construct_prime_step_sets():
    """Construct sets using prime step sizes (non-superincreasing)."""
    candidates = []
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

    for n in range(3, 10):
        for seed in range(10):
            random.seed(seed * 777 + n)
            steps = [random.choice(primes) for _ in range(n - 1)]
            U = [0]
            curr = 0
            for s in steps:
                curr += s
                U.append(curr)
            if len(set(U)) == len(U):
                candidates.append(sorted(U))

    # Also try non-prime but sparse steps
    for n in range(3, 10):
        for seed in range(10):
            random.seed(seed * 999 + n + 1)
            steps = [random.randint(2, 20) for _ in range(n - 1)]
            U = [0]
            curr = 0
            for s in steps:
                curr += s
                U.append(curr)
            if len(set(U)) == len(U):
                candidates.append(sorted(U))

    return candidates

def _exhaustive_small(max_n=6, max_val=40, deadline=None):
    """Exhaustive search for small sets."""
    from itertools import combinations
    best_score = 0.0
    best_U = [0, 1, 3]

    for n in range(2, max_n + 1):
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
            U = [0] + list(combo)
            U = sorted(U)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]

    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin for 150s limit

    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)

    # 1. Exhaustive search for small sets
    try:
        if time.time() < deadline:
            U_sol, s_sol = _exhaustive_small(max_n=5, max_val=35, deadline=deadline)
            if s_sol > best_score:
                best_score = s_sol
                best_U = U_sol
    except Exception:
        pass

    # 2. Generate diverse candidates
    all_candidates = []

    # Golomb-inspired
    golomb_sets = _construct_golomb_inspired()
    all_candidates.extend(golomb_sets)

    # Arithmetic perturbations
    arith_sets = _construct_arithmetic_perturbations()
    all_candidates.extend(arith_sets)

    # Prime step sets
    prime_sets = _construct_prime_step_sets()
    all_candidates.extend(prime_sets)

    # Evaluate all candidates
    for U in all_candidates:
        if time.time() > deadline:
            break
        s = _score_set_fast(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # 3. Local search on best found
    if time.time() < deadline - 5:
        opt_U, opt_score = _local_search(best_U, max_iters=400, deadline=deadline - 5, move_radius=25)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    # 4. Simulated annealing from best
    if time.time() < deadline - 3:
        sa_U, sa_score = _simulated_annealing(best_U, max_iters=300, deadline=deadline - 3)
        if sa_score > best_score:
            best_score = sa_score
            best_U = sa_U

    # 5. Final local search polish
    if time.time() < deadline - 2:
        final_U, final_score = _local_search(best_U, max_iters=200, deadline=deadline - 2, move_radius=15)
        if final_score > best_score:
            best_score = final_score
            best_U = final_U

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
