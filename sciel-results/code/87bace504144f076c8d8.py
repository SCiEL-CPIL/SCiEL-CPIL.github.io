import time
import random
from itertools import combinations

def _max_sidon_size(arr):
    """Find the largest subset of arr that is Sidon (all pair sums a_i+a_j for i<=j distinct)."""
    n = len(arr)
    best = 1
    # Try all subsets from largest to smallest
    for k in range(n, 0, -1):
        found = False
        for subset in combinations(range(n), k):
            if _is_sidon([arr[i] for i in subset]):
                best = k
                found = True
                break
        if found:
            break
    return best

def _is_sidon(arr):
    """Check if arr is a Sidon set (all sums a_i + a_j for i <= j are distinct)."""
    sums = set()
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s
    
    best_set = list(range(12))
    best_score = _max_sidon_size(best_set)
    
    # Try some structured sets
    candidates = []
    
    # Arithmetic progressions
    for start in range(0, 128 - 11 * 10 + 1):
        for step in range(1, 12):
            arr = [start + i * step for i in range(12)]
            if max(arr) <= 128 and len(set(arr)) == 12:
                candidates.append(arr)
    
    # Clusters
    for base in range(0, 100):
        for cluster_size in range(2, 6):
            arr = []
            for c in range(0, 12, cluster_size):
                arr.extend(range(base + c, base + c + cluster_size))
            if len(arr) == 12 and max(arr) <= 128:
                candidates.append(arr)
    
    # Random sets
    random.seed(42)
    for _ in range(5000):
        arr = sorted(random.sample(range(129), 12))
        candidates.append(arr)
    
    # Evaluate candidates
    for arr in candidates:
        if time.time() > deadline - 5:
            break
        score = _max_sidon_size(arr)
        if score < best_score:
            best_score = score
            best_set = arr
    
    # Local search: try to improve by swapping elements
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current = list(best_set)
        current_score = best_score
        for i in range(12):
            if time.time() > deadline - 5:
                break
            for new_val in range(129):
                if new_val in current:
                    continue
                old_val = current[i]
                current[i] = new_val
                current.sort()
                score = _max_sidon_size(current)
                if score < current_score:
                    current_score = score
                    best_set = list(current)
                    best_score = score
                    improved = True
                current[i] = old_val
                current.sort()
    
    return best_set
