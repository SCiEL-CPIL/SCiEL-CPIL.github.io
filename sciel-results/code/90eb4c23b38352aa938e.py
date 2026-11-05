import random
import time
from itertools import combinations

def is_sidon(subset):
    """
    Check if a given subset has all unique unordered pair sums including diagonals.
    Sums a[i] + a[j] for i <= j must be distinct.
    """
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

def max_sidon_size(candidate_set):
    """
    Find the size of the largest Sidon subset of the given candidate set.
    Returns an integer k.
    """
    n = len(candidate_set)
    # Check from largest possible subset size down
    for k in range(n, 0, -1):
        # If k is large, few combinations. If k is small, many, but we stop early if found.
        # We want to know the MAX size. So we check k=n, n-1, ...
        # If we find a Sidon subset of size k, we return k.
        for subset in combinations(candidate_set, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    # 90s limit, use 80s to be safe
    deadline = start_time + 80.0
    
    # Initial candidates: Structured sets that likely have many sum collisions
    # APs, dense clusters, etc.
    initial_candidates = [
        list(range(12)), # 0..11
        list(range(0, 24, 2)), # Even 0..22
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32], # Blocks
        [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23], # Tighter blocks
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], # Even
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15], # Two blocks
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13], # Two blocks
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], # Perturbed AP
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12], # Perturbed AP
    ]
    
    best_set = None
    best_size = 13 # Max possible is 12
    
    # Evaluate initial candidates
    for c in initial_candidates:
        if len(set(c)) != 12:
            continue
        if any(x < 0 or x > 128 for x in c):
            continue
        sz = max_sidon_size(c)
        if sz < best_size:
            best_size = sz
            best_set = list(c)
            if best_size <= 2:
                break
    
    # If no valid candidate found (should not happen), fallback
    if best_set is None:
        best_set = list(range(12))
        best_size = max_sidon_size(best_set)
        
    current_set = list(best_set)
    current_size = best_size
    
    random.seed(42)
    
    # Local Search / Hill Climbing
    # We want to minimize current_size.
    # Strategy: Randomly pick an index, change the value to a random int in [0, 128]
    # ensuring uniqueness. If the new set has a smaller max_sidon_size, accept.
    
    while time.time() < deadline:
        # Try a batch of mutations
        for _ in range(100):
            if time.time() >= deadline:
                break
            
            new_set = list(current_set)
            idx = random.randint(0, 11)
            
            # Generate a new value not in the set
            # To make it faster, we can just pick a random value and check
            # If collision, pick another.
            # Since range is 129 and set size 12, collision prob is ~10%.
            attempts = 0
            while attempts < 10:
                new_val = random.randint(0, 128)
                if new_val not in new_set:
                    break
                attempts += 1
            
            if attempts == 10:
                continue # Couldn't find a unique value, skip (rare)
                
            new_set[idx] = new_val
            
            # Evaluate new set
            new_size = max_sidon_size(new_set)
            
            if new_size < current_size:
                current_set = new_set
                current_size = new_size
                if current_size < best_size:
                    best_size = current_size
                    best_set = list(current_set)
                    if best_size <= 2:
                        # Can't get lower than 2? 
                        # A set of 3 elements {a,b,c} is Sidon if a+a, a+b, a+c, b+b, b+c, c+c are unique.
                        # It is possible to have max size 2 if no triple is Sidon.
                        # It is possible to have max size 1? No, any single element is Sidon.
                        # Any pair {a,b}: sums a+a, a+b, b+b. Always unique unless a=0, b=0? No distinct.
                        # So min size is 2.
                        return best_set
            elif new_size == current_size and random.random() < 0.01:
                # Accept equal moves with low probability to escape local minima
                current_set = new_set
                # current_size unchanged
                
    return best_set
