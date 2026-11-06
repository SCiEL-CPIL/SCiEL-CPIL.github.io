"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _max_sidon_subset_size(arr):
    """Compute the largest subset of arr with all pair sums unique (including diagonals)."""
    n = len(arr)
    # Try from largest possible down
    for k in range(n, 0, -1):
        for combo in itertools.combinations(range(n), k):
            # Check if this subset is a Sidon set
            sums = set()
            valid = True
            for i in range(k):
                for j in range(i, k):
                    s = arr[combo[i]] + arr[combo[j]]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                return k
    return 1

def _evaluate_set(arr):
    """Evaluate the max Sidon subset size for a given set."""
    return _max_sidon_subset_size(arr)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    best_score = 12
    best_set = list(range(12))
    
    # Strategy: Find 12 numbers that maximize sum collisions
    # Key insight: if we pick numbers so that many pair sums are equal,
    # the largest Sidon subset will be small.
    
    # Approach 1: Concentrate in small range [0, n]
    # For 12 numbers in [0, 11], pair sums range [0, 22], only 23 values
    # This forces massive collisions
    
    # Let's search for good sets
    # First, try some structured candidates
    
    candidates = []
    
    # Dense small range
    for offset in range(0, 20):
        candidates.append(list(range(offset, offset + 12)))
    
    # Symmetric pairs (a, 2c-a) create sum collisions
    # E.g., pairs summing to same value
    for c in range(0, 130):
        # Pick pairs around c
        vals = set()
        for d in range(0, 20):
            if c - d >= 0:
                vals.add(c - d)
            if c + d <= 128:
                vals.add(c + d)
            if len(vals) >= 12:
                break
        if len(vals) == 12:
            candidates.append(sorted(vals))
    
    # Random sets in small ranges
    random.seed(42)
    for _ in range(500):
        lo = random.randint(0, 50)
        hi = random.randint(lo + 11, min(lo + 30, 128))
        candidates.append(sorted(random.sample(range(lo, hi + 1), 12)))
    
    # Random sets in medium ranges
    for _ in range(500):
        candidates.append(sorted(random.sample(range(129), 12)))
    
    # Arithmetic progressions
    for start in range(0, 100):
        for d in range(1, 15):
            arr = [start + i * d for i in range(12)]
            if max(arr) <= 128:
                candidates.append(arr)
    
    # Evaluate candidates
    evaluated = 0
    for arr in candidates:
        if time.time() > deadline:
            break
        if len(set(arr)) != 12:
            continue
        if any(x < 0 or x > 128 for x in arr):
            continue
        score = _evaluate_set(arr)
        evaluated += 1
        if score < best_score:
            best_score = score
            best_set = list(arr)
            if best_score <= 3:
                break  # Can't do much better than 3
    
    # If we have time, do a more targeted search
    if time.time() < deadline - 10 and best_score > 3:
        # Local search: perturb best set
        for _ in range(1000):
            if time.time() > deadline:
                break
            new_set = list(best_set)
            # Randomly replace 1-3 elements
            num_replace = random.randint(1, 3)
            for _ in range(num_replace):
                idx = random.randint(0, 11)
                new_set[idx] = random.randint(0, 128)
            if len(set(new_set)) != 12:
                continue
            score = _evaluate_set(new_set)
            if score < best_score:
                best_score = score
                best_set = new_set
                if best_score <= 3:
                    break
    
    # Verify final answer
    final_set = sorted(best_set)
    assert len(final_set) == 12
    assert all(0 <= x <= 128 for x in final_set)
    assert len(set(final_set)) == 12
    
    return final_set
# EVOLVE-BLOCK-END
