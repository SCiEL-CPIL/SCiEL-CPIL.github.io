import random
import time
import itertools

def is_sidon(subset):
    """Check if a subset is Sidon (all pair sums including diagonals are unique)."""
    n = len(subset)
    if n <= 1:
        return True
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def is_sidon_4(a, b, c, d):
    """Fast check for 4-element Sidon subset."""
    s = (2*a, a+b, a+c, a+d, 2*b, b+c, b+d, 2*c, c+d, 2*d)
    return len(set(s)) == 10

def count_sidon_4_subsets(s):
    """Count number of Sidon 4-subsets in set s (12 elements -> 495 combos)."""
    count = 0
    n = len(s)
    for i in range(n - 3):
        a = s[i]
        for j in range(i + 1, n - 2):
            b = s[j]
            for k in range(j + 1, n - 1):
                c = s[k]
                for l in range(k + 1, n):
                    d = s[l]
                    if is_sidon_4(a, b, c, d):
                        count += 1
    return count

def largest_sidon_subset_size(s):
    """Find the size of the largest Sidon subset of s exactly."""
    s = sorted(s)
    n = len(s)
    # Check sizes from n down to 3
    for size in range(n, 2, -1):
        for combo in itertools.combinations(range(n), size):
            subset = [s[i] for i in combo]
            if is_sidon(subset):
                return size
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit

    best_set = None
    best_score = float('inf')

    def update_best(s, score):
        nonlocal best_set, best_score
        if score < best_score:
            best_score = score
            best_set = list(s)
            if best_score <= 3:  # Can't do better than 3
                return True
        return False

    random.seed(42)

    # Try structured initializations with fast surrogate
    candidates = [
        list(range(12)),
        [0, 1, 3, 7, 15, 31, 63, 127, 2, 4, 8, 16],
        [0, 1, 2, 4, 8, 16, 32, 64, 5, 10, 20, 40],
        [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66],
        [0, 1, 2, 5, 7, 13, 18, 26, 35, 46, 58, 71],
    ]

    for cand in candidates:
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            c4 = count_sidon_4_subsets(cand)
            if c4 == 0:
                # Verify with exact evaluator
                exact = largest_sidon_subset_size(cand)
                if update_best(cand, exact):
                    return best_set
            else:
                # Surrogate says > 3, but let's still track
                # We'll use the count as a proxy for ranking during search
                pass

    # Conflict-graph inspired heuristic:
    # For each candidate integer, compute how many "sum collisions" it creates
    # with other pairs. We want to pick integers that maximize collisions.
    # This is approximated by: for a set S, count how many 4-subsets are NON-Sidon
    # (i.e., have sum collisions). We want to MINIMIZE Sidon 4-subsets.

    # Main search: simulated annealing with fast surrogate
    T = 1.0
    cooling_rate = 0.995
    min_T = 0.001

    for restart in range(500):
        if time.time() > deadline:
            break

        # Random initialization
        current_set = sorted(random.sample(range(129), 12))
        current_score = count_sidon_4_subsets(current_set)

        # If surrogate is 0, verify exactly
        if current_score == 0:
            exact = largest_sidon_subset_size(current_set)
            if update_best(current_set, exact):
                return best_set

        # Local search
        T = 1.0
        no_improve = 0
        while T > min_T and time.time() < deadline and no_improve < 80:
            # Generate neighbor: replace one or two elements
            new_set = list(current_set)
            num_replacements = random.choice([1, 1, 2])

            for _ in range(num_replacements):
                idx = random.randint(0, 11)
                old_val = new_set[idx]
                if random.random() < 0.5:
                    offset = random.randint(-5, 5)
                    new_val = max(0, min(128, old_val + offset))
                else:
                    new_val = random.randint(0, 128)

                if new_val != old_val and new_val not in new_set:
                    new_set[idx] = new_val
                    new_set.sort()

            if len(set(new_set)) != 12:
                no_improve += 1
                continue

            new_score = count_sidon_4_subsets(new_set)
            delta = new_score - current_score

            if delta < 0 or random.random() < pow(2.718, -delta / max(T, 1e-9)):
                current_set = new_set
                current_score = new_score
                no_improve = 0

                if current_score == 0:
                    exact = largest_sidon_subset_size(current_set)
                    if update_best(current_set, exact):
                        return best_set
            else:
                no_improve += 1

            T *= cooling_rate

    # Final verification of best_set if we haven't returned yet
    if best_set is None:
        # Fallback: return a reasonable set
        best_set = list(range(12))

    return best_set
