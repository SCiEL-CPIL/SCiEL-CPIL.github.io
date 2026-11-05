import random
import time

def is_sidon(subset):
    """Check if a given subset has all unique pair sums including diagonals."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(candidate_set):
    """Find the size of the largest Sidon subset of the given candidate set."""
    n = len(candidate_set)
    # Check from largest possible subset size down
    # We only need to check up to size n, but for n=12, we can be efficient
    # C(12, k) is small for k near 12 or small.
    # However, if the max size is small (e.g., 4), checking size 12 down to 5 is expensive?
    # C(12,12)=1, C(12,11)=12, C(12,10)=66, C(12,9)=220, C(12,8)=495, C(12,7)=792, C(12,6)=924
    # Total checks to prove max size < 7 is ~ 1+12+66+220+495 = 794.
    # To prove max size = 4, we check 12..5 (1+12+66+220+495 = 794) and then find one in size 4.
    # This is very fast.
    
    from itertools import combinations
    
    for k in range(n, 0, -1):
        for subset in combinations(candidate_set, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds to be safe
    
    best_set = list(range(12))
    best_size = max_sidon_size(best_set)
    
    # Initial deterministic checks with structured sets
    candidates = [
        list(range(12)),
        [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32],
        [0, 1, 3, 7, 12, 18, 25, 33, 42, 52, 63, 75], # Random-ish
        [0, 1, 4, 10, 20, 40, 80, 2, 5, 15, 35, 75],
    ]
    
    for c in candidates:
        if len(set(c)) != 12: continue
        if any(x < 0 or x > 128 for x in c): continue
        sz = max_sidon_size(c)
        if sz < best_size:
            best_size = sz
            best_set = c
    
    # Random search
    random.seed(42)
    
    while time.time() < deadline:
        # Generate random set of 12 distinct ints in [0, 128]
        s = random.sample(range(129), 12)
        sz = max_sidon_size(s)
        if sz < best_size:
            best_size = sz
            best_set = s
            if best_size == 3: # Minimum possible?
                # Can we go lower? Size 1 or 2?
                # A set of 12 elements always has a Sidon subset of size at least...
                # By Erdos-Turan, but for small N, likely 3 or 4.
                # If we find 3, it's likely optimal.
                break
    
    return best_set
