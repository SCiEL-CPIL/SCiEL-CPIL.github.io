import random
import time
from itertools import combinations

def is_sidon_fast(arr):
    """Check if arr is a Sidon set (all pair sums including diagonals unique)."""
    sums = set()
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the largest subset of S that is Sidon."""
    n = len(S)
    for k in range(n, 0, -1):
        for comb in combinations(S, k):
            if is_sidon_fast(comb):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin before 90s kill

    random.seed(42)

    # Start with consecutive integers
    best_set = list(range(12))
    best_size = max_sidon_size(best_set)

    # Phase 1: Random sampling in small ranges to find dense clusters
    # Smaller diameter => more sum collisions => smaller max Sidon subset
    for _ in range(20000):
        if time.time() > deadline:
            break

        # Try different range sizes
        r = random.choice([20, 25, 30, 35, 40, 50])
        s = random.sample(range(r), 12)
        s.sort()

        size = max_sidon_size(s)

        if size < best_size:
            best_size = size
            best_set = s

            if best_size <= 3:
                break

    # Phase 2: Local search from best found
    current_set = best_set[:]
    current_size = best_size

    for _ in range(30000):
        if time.time() > deadline:
            break

        # Pick a random element to perturb
        idx = random.randint(0, 11)
        old_val = current_set[idx]

        # Try nearby values to maintain density
        offset = random.randint(-5, 5)
        new_val = old_val + offset

        # Also try random values in a small window
        if random.random() < 0.3:
            new_val = random.randint(max(0, old_val - 10), min(128, old_val + 10))

        if new_val in current_set:
            continue

        new_set = current_set.copy()
        new_set[idx] = new_val
        new_set.sort()

        new_size = max_sidon_size(new_set)

        if new_size < current_size:
            current_size = new_size
            current_set = new_set
            best_size = current_size
            best_set = new_set

            if best_size <= 3:
                break
        elif new_size == current_size and random.random() < 0.05:
            # Small probability of neutral move to escape local minima
            current_set = new_set

    return best_set
