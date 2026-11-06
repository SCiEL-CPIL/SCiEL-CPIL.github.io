"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def _is_sidon(subset):
    """Check if a tuple of integers is a Sidon set (including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_subset_size(arr):
    """Find the largest subset of arr where all pair sums (i<=j) are unique."""
    n = len(arr)
    # Try subsets from largest to smallest
    for size in range(n, 0, -1):
        found = False
        for subset in itertools.combinations(arr, size):
            if _is_sidon(subset):
                return size
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    # Helper to get a random valid candidate in a specific range
    def random_candidate(max_val=25):
        # Ensure 12 distinct integers in [0, max_val]
        # max_val must be at least 11
        return random.sample(range(0, max_val + 1), 12)

    # Initial candidates
    candidates = [
        list(range(12)),
        list(range(0, 24, 2)),
        [0, 1, 2, 3, 4, 5, 122, 123, 124, 125, 126, 127],
        [i * 10 + 5 for i in range(12)],
        [0, 1, 3, 7, 15, 31, 63, 94, 95, 97, 101, 111],
        list(range(0, 128, 11))[:12],
        [0, 1, 2, 10, 11, 12, 116, 117, 118, 126, 127, 128],
        [0, 1, 2, 4, 8, 16, 32, 64, 65, 66, 68, 72],
    ]
    
    # Add some random candidates in small ranges
    random.seed(42)
    for _ in range(20):
        candidates.append(random_candidate(25))
    for _ in range(20):
        candidates.append(random_candidate(30))
    for _ in range(10):
        candidates.append(random_candidate(40))

    best_set = None
    best_size = 99

    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        size = _max_sidon_subset_size(cand)
        if size < best_size:
            best_size = size
            best_set = cand[:]
            if best_size <= 4:
                # Can't do much better than 4 easily, but let's see if we can find 3?
                # 3 is theoretically possible but very hard.
                # Let's continue searching briefly if time permits.
                pass

    # Local Search
    # We want to minimize best_size.
    # If best_size is 5, we try to find a set with size 4.
    # If best_size is 4, we try to find a set with size 3.
    
    if best_set is None:
        best_set = list(range(12))
        best_size = _max_sidon_subset_size(best_set)

    current_set = best_set[:]
    current_size = best_size

    # If we already have 4, we might try to improve to 3.
    # If we have 5, we try to improve to 4.
    
    search_steps = 5000
    for step in range(search_steps):
        if time.time() > deadline - 5:
            break
        
        # Pick a random index to mutate
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Try a new value. Bias towards small numbers to increase collisions.
        # Range 0-30 is a good sweet spot.
        new_val = random.randint(0, 30)
        
        if new_val == old_val:
            continue
            
        # Check if new_val is already in the set
        if new_val in current_set:
            continue
            
        # Apply mutation
        current_set[idx] = new_val
        
        # Evaluate
        new_size = _max_sidon_subset_size(current_set)
        
        if new_size < current_size:
            # Improvement
            current_size = new_size
            if new_size < best_size:
                best_size = new_size
                best_set = current_set[:]
                # If we reached 4, we are happy. If 3, even better.
                if best_size <= 3:
                    break
        else:
            # Revert
            current_set[idx] = old_val

    # Final check
    if best_set is None:
        return list(range(12))
        
    return best_set
# EVOLVE-BLOCK-END
