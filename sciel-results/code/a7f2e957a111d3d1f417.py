"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def _is_sidon(arr):
    """Check if arr has all pair sums (i<=j) unique."""
    n = len(arr)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_subset_size(arr, timeout=5.0):
    """Find the largest subset of arr with all pair sums (i<=j) unique."""
    n = len(arr)
    arr = sorted(arr)
    start = time.time()
    for size in range(n, 0, -1):
        if time.time() - start > timeout:
            # Return a conservative upper bound
            return size
        for subset in itertools.combinations(range(n), size):
            sub = [arr[i] for i in subset]
            if _is_sidon(sub):
                return size
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    best_score = 12
    best_arr = list(range(12))
    
    # Evaluate seed
    seed_arr = list(range(12))
    best_score = _max_sidon_subset_size(seed_arr)
    best_arr = seed_arr[:]
    
    # Try structured candidates
    candidates = []
    for start in range(0, 15):
        arr = list(range(start, start + 12))
        candidates.append(arr)
    
    # Add more structured candidates with collisions
    for k in range(12, 5, -1):
        arr = list(range(k)) + [k + 5, k + 10, k + 15, k + 20][:12-k]
        if len(arr) == 12 and max(arr) <= 128:
            candidates.append(arr)
    
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 30])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 30, 40])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 20, 30, 40, 50, 60])
    candidates.append([0, 1, 2, 3, 4, 5, 20, 30, 40, 50, 60, 70])
    candidates.append([0, 1, 2, 3, 4, 20, 30, 40, 50, 60, 70, 80])
    candidates.append([0, 1, 2, 3, 20, 30, 40, 50, 60, 70, 80, 90])
    candidates.append([0, 1, 2, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    candidates.append([0, 1, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
    candidates.append([0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 20])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 15, 20, 25, 30])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 15, 20, 25, 30, 35])
    candidates.append([0, 1, 2, 3, 4, 5, 15, 20, 25, 30, 35, 40])
    candidates.append([0, 1, 2, 3, 4, 15, 20, 25, 30, 35, 40, 45])
    candidates.append([0, 1, 2, 3, 15, 20, 25, 30, 35, 40, 45, 50])
    candidates.append([0, 1, 2, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    candidates.append([0, 1, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    candidates.append([0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    
    for arr in candidates:
        if len(set(arr)) != 12 or min(arr) < 0 or max(arr) > 128:
            continue
        score = _max_sidon_subset_size(arr)
        if score < best_score:
            best_score = score
            best_arr = arr[:]
    
    # Local search
    random.seed(42)
    while time.time() < deadline:
        # Perturb best
        arr = best_arr[:]
        for _ in range(random.randint(1, 2)):
            idx = random.randint(0, 11)
            arr[idx] = random.randint(0, 128)
        
        if len(set(arr)) != 12:
            continue
        arr = sorted(arr)
        
        score = _max_sidon_subset_size(arr)
        if score < best_score:
            best_score = score
            best_arr = arr[:]
            if best_score <= 4:
                break
    
    return best_arr
# EVOLVE-BLOCK-END
