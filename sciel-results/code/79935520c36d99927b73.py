"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    def is_sidon(subset):
        """Check if subset has all pair sums unique (including diagonals)."""
        sums = set()
        n = len(subset)
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def max_sidon_size(arr):
        """Find the largest Sidon subset size in arr."""
        n = len(arr)
        # Try from largest to smallest
        for size in range(n, 0, -1):
            for subset in itertools.combinations(arr, size):
                if is_sidon(list(subset)):
                    return size
        return 1
    
    def score_candidate(arr):
        """Lower is better: the max Sidon subset size."""
        return max_sidon_size(arr)
    
    # Initial candidate: clustered to create sum collisions
    # Strategy: use pairs that sum to same values
    # E.g., pairs summing to 20: (0,20),(1,19),(2,18),(3,17),(4,16),(5,15)
    # Plus some more to make 12
    
    # Let's try various structured candidates
    candidates = []
    
    # Pattern 1: Two clusters
    candidates.append([0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25])
    # Pattern 2: Pairs summing to same constant
    candidates.append([0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20])
    # Pattern 3: Arithmetic progression (bad - large Sidon)
    candidates.append(list(range(12)))
    # Pattern 4: More clustered
    candidates.append([0, 0+1, 0+2, 0+3, 0+4, 0+5, 10, 11, 12, 13, 14, 15])
    
    # Best known so far
    best_candidate = None
    best_score = 12  # worst case
    
    for cand in candidates:
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        s = score_candidate(cand)
        if s < best_score:
            best_score = s
            best_candidate = cand[:]
    
    # Local search: try swapping elements to reduce max Sidon size
    if best_candidate is None:
        best_candidate = list(range(12))
        best_score = 12
    
    # Try random perturbations
    random.seed(42)
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Start from best and make a small change
        cand = best_candidate[:]
        
        # Randomly swap one element with another value in [0,128]
        idx = random.randint(0, 11)
        old_val = cand[idx]
        new_val = random.randint(0, 128)
        
        if new_val in cand and new_val != old_val:
            continue
        
        cand[idx] = new_val
        
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        
        s = score_candidate(cand)
        if s < best_score:
            best_score = s
            best_candidate = cand[:]
    
    return best_candidate
# EVOLVE-BLOCK-END
