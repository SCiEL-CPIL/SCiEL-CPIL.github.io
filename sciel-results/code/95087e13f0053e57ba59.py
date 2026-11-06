"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if subset has all unordered pair sums (including diagonals) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(arr):
    """Find the largest Sidon subset size in arr."""
    n = len(arr)
    # Try from largest possible down
    for size in range(n, 1, -1):
        for combo in combinations(arr, size):
            if _is_sidon(list(combo)):
                return size
    return 2  # Any 2 distinct elements form a Sidon set

def _max_sidon_size_fast(arr):
    """Faster check using early termination."""
    n = len(arr)
    # Upper bound: for a Sidon set of size k in [0,M], need k(k+1)/2 <= M+1
    # So k <= ~sqrt(2*M)
    # For our purposes, just do greedy + verification
    # First, try if whole set is Sidon
    if _is_sidon(arr):
        return n
    # Try n-1
    for size in range(n-1, 3, -1):
        found = False
        for combo in combinations(arr, size):
            if _is_sidon(list(combo)):
                return size
    # Check size 4
    for combo in combinations(arr, 4):
        if _is_sidon(list(combo)):
            return 4
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    # Start with a good initial set - consecutive integers give max Sidon size ~4-5
    best_set = list(range(12))
    best_score = _max_sidon_size_fast(best_set)
    
    # Try some structured sets
    candidate_sets = [
        list(range(12)),           # 0-11
        [0,1,2,3,4,5,6,7,8,9,10,11],
        [0,1,2,3,4,5,6,7,8,9,10,20],  # spread one out
        [0,1,2,3,4,5,6,7,8,9,15,20],
        [0,1,2,3,4,5,6,7,8,12,16,20],
        [0,1,2,3,4,5,6,7,10,14,18,22],
        [0,1,2,3,4,5,6,10,14,18,22,26],
        [0,1,2,3,4,8,12,16,20,24,28,32],
        [0,1,2,4,6,8,10,12,14,16,18,20],  # even-heavy
        [0,1,3,5,7,9,11,13,15,17,19,21],  # odd-heavy
    ]
    
    for cs in candidate_sets:
        if time.time() > deadline:
            break
        if len(set(cs)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cs):
            continue
        score = _max_sidon_size_fast(cs)
        if score < best_score:
            best_score = score
            best_set = list(cs)
    
    # Local search: try swapping elements
    random.seed(42)
    current = list(best_set)
    current_score = best_score
    
    while time.time() < deadline:
        # Try random swaps
        improved = False
        for _ in range(100):
            if time.time() > deadline:
                break
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            old_val = current[i]
            current[i] = new_val
            new_score = _max_sidon_size_fast(current)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_set = list(current)
                improved = True
            else:
                current[i] = old_val
        
        if not improved:
            # Try adding/removing structure
            # Perturb more aggressively
            for _ in range(50):
                if time.time() > deadline:
                    break
                # Replace 2-3 elements
                for _ in range(random.randint(2, 3)):
                    i = random.randint(0, 11)
                    new_val = random.randint(0, 128)
                    if new_val not in current:
                        current[i] = new_val
                if len(set(current)) == 12:
                    new_score = _max_sidon_size_fast(current)
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_set = list(current)
    
    return best_set
# EVOLVE-BLOCK-END
