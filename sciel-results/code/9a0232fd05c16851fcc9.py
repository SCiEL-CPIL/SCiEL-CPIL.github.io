"""Find a 12-element set in [0,128] minimizing the largest Sidon subset (with diagonals)."""
import random
import time
from itertools import combinations

def _is_sidon_with_diag(subset):
    """Check if a sorted list has all pair sums (i<=j) unique."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size_exhaustive(arr):
    """Exact: largest subset of arr with all pair sums (i<=j) unique."""
    n = len(arr)
    # Try from size 6 down; C(12,6)=924, C(12,5)=792, C(12,4)=495
    # We only need to check down to where we find one
    for k in range(n, 3, -1):  # size 1 and 2 are always Sidon
        for combo in combinations(arr, k):
            if _is_sidon_with_diag(list(combo)):
                return k
    return 3  # if no size 4+, answer is 3 (or less, but 3 is minimum possible for distinct ints)

def construct():
    deadline = time.time() + 58  # safety margin before 60s limit
    best_set = list(range(12))
    best_size = _max_sidon_size_exhaustive(best_set)
    
    random.seed(42)
    
    # Pre-compute: what's the max Sidon size for [0,11]?
    # Let's verify and use as baseline
    
    def evaluate(s):
        """Return the exact max Sidon subset size for set s."""
        return _max_sidon_size_exhaustive(s)
    
    # Strategy: Focus on dense sets with small range
    # A set with small range has more sum collisions
    
    # Phase 1: Try structured dense sets
    candidates = []
    
    # Consecutive blocks of various lengths
    for start in range(0, 118):  # start such that start+11 <= 128
        candidates.append(list(range(start, start + 12)))
    
    # Sets with small gaps (dense but not consecutive)
    # Try sets where elements are clustered
    for base in range(0, 120):
        # Small perturbations of consecutive
        for _ in range(50):
            s = list(range(base, base + 12))
            # Perturb a few elements slightly
            for _ in range(random.randint(0, 3)):
                idx = random.randint(0, 11)
                offset = random.randint(-2, 2)
                new_val = s[idx] + offset
                if 0 <= new_val <= 128 and new_val not in s:
                    s[idx] = new_val
            if len(set(s)) == 12:
                candidates.append(s)
    
    # Evaluate structured candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        sz = evaluate(cand)
        if sz < best_size:
            best_size = sz
            best_set = list(cand)
            if best_size <= 4:
                # Try to find even better
                pass
    
    # Phase 2: Local search from best
    current = list(best_set)
    current_size = best_size
    
    # Local search: try small perturbations
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Create perturbed copy
        A_new = list(current)
        idx = random.randint(0, 11)
        
        # Biased toward small steps to maintain density
        if random.random() < 0.6:
            step = random.randint(-3, 3)
            new_val = A_new[idx] + step
        else:
            # Jump within a small window around current range
            min_val = min(current)
            max_val = max(current)
            window = max(5, (max_val - min_val) // 3)
            new_val = random.randint(max(0, min_val - window), min(128, max_val + window))
        
        if new_val < 0 or new_val > 128 or new_val in A_new:
            continue
        
        A_new[idx] = new_val
        new_size = evaluate(A_new)
        
        if new_size < current_size:
            current_size = new_size
            current = A_new
            if current_size < best_size:
                best_size = current_size
                best_set = list(current)
                if best_size <= 3:
                    return best_set
        elif new_size == current_size:
            # Accept with probability to explore plateau
            if random.random() < 0.2:
                current = A_new
        else:
            # Revert (already have original in current)
            pass
        
        # Periodic random restart from best
        if iteration % 2000 == 1999:
            # Perturb best_set more aggressively
            current = list(best_set)
            for _ in range(3):
                idx = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val not in current:
                    current[idx] = new_val
    
    return best_set
