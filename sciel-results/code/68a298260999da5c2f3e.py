import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if all pair sums (including diagonals) are unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(candidates):
    """Find the largest Sidon subset of candidates."""
    n = len(candidates)
    for k in range(n, 0, -1):
        for subset in combinations(candidates, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_set = None
    best_size = 13  # worst case
    
    # Start with some structured candidates
    # Try consecutive ranges
    for start in range(0, 117):  # 0 to 116, so start+11 <= 127
        if time.time() > deadline:
            break
        candidate = list(range(start, start + 12))
        ms = max_sidon_size(candidate)
        if ms < best_size:
            best_size = ms
            best_set = candidate
            if best_size <= 4:
                break
    
    # Random search
    random.seed(42)
    while time.time() < deadline:
        candidate = random.sample(range(129), 12)
        ms = max_sidon_size(candidate)
        if ms < best_size:
            best_size = ms
            best_set = candidate
            if best_size <= 4:
                break
    
    return best_set
