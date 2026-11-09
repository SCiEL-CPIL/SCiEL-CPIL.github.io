"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if a subset has unique pair sums (a+b for a<=b)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        a = subset[i]
        for j in range(i, n):
            s = a + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    n = len(S)
    # Check from largest size down
    for k in range(n, 0, -1):
        for comb in combinations(S, k):
            if is_sidon(comb):
                return k
    return 0

def count_sidon_4_subsets(arr):
    """Count how many 4-element subsets are Sidon. Returns 0 if none."""
    count = 0
    for sub4 in combinations(arr, 4):
        if is_sidon(sub4):
            count += 1
    return count

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 75.0  # Keep 15s buffer before 90s limit
    
    # Initialize with the simple AP which gives max Sidon = 4
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)  # should be 4
    
    # Try to find a set with max Sidon <= 3
    # Strategy: use sets with many 3-term APs or dense clustering
    random.seed(42)
    
    # Try structured sets first - these are designed to have many pair-sum collisions
    structured_candidates = [
        # Consecutive integers - many collisions
        list(range(12)),
        # Pairs that create many sum collisions
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # Arithmetic progression with step 1
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # Two interleaved APs
        [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],
        # Clustered pairs
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        # Dense small range
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # Spread but with collisions
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
    ]
    
    for cand in structured_candidates:
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            s = max_sidon_size(cand)
            if s < best_score:
                best_score = s
                best_set = cand[:]
                if best_score <= 3:
                    break
    
    # If we already found max Sidon = 3, we're done
    if best_score <= 3:
        return best_set
    
    # Random search for max_sidon <= 3
    # Use a smarter approach: generate sets with many small gaps to create sum collisions
    for attempt in range(50000):
        if time.time() - start_time > deadline:
            break
        
        # Strategy: create sets with many close pairs to force sum collisions
        # Use a biased random generation
        if attempt % 3 == 0:
            # Generate from small range to force collisions
            base = random.randint(0, 30)
            offsets = random.sample(range(0, 15), 12)
            arr = sorted(set(base + o for o in offsets))
            if len(arr) < 12:
                # Fill in remaining
                remaining = [x for x in range(base, base + 20) if x not in arr]
                random.shuffle(remaining)
                arr.extend(remaining[:12 - len(arr)])
                arr = sorted(set(arr))
                if len(arr) != 12:
                    continue
        elif attempt % 3 == 1:
            # Generate with clustering
            clusters = []
            for _ in range(4):
                center = random.randint(0, 100)
                cluster = [center + random.randint(-2, 2) for _ in range(3)]
                clusters.extend(cluster)
            arr = sorted(set(clusters))
            if len(arr) < 12:
                remaining = [x for x in range(0, 129) if x not in arr]
                random.shuffle(remaining)
                arr.extend(remaining[:12 - len(arr)])
                arr = sorted(set(arr))
                if len(arr) != 12:
                    continue
        else:
            # Standard random
            arr = sorted(random.sample(range(129), 12))
        
        # Quick check: count Sidon 4-subsets
        sidon4_count = count_sidon_4_subsets(arr)
        
        if sidon4_count == 0:
            # Verify it's really max Sidon <= 3
            actual_max = max_sidon_size(arr)
            if actual_max < best_score:
                best_score = actual_max
                best_set = arr[:]
                if best_score <= 3:
                    break
    
    return best_set
# EVOLVE-BLOCK-END
