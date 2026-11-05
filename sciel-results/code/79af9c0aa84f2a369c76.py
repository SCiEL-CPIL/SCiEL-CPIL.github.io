"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter
from itertools import combinations

def _is_sidon(vals):
    """Check if vals has all unordered pair sums unique (including diagonals)."""
    n = len(vals)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = vals[i] + vals[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _count_duplicate_sums(subset):
    """Count how many unordered pair sums (including diagonals) are duplicated."""
    n = len(subset)
    sums = [subset[i] + subset[j] for i in range(n) for j in range(i, n)]
    c = Counter(sums)
    return sum(k * (k - 1) // 2 for k in c.values())

def _max_sidon_subset_size(subset, deadline=None):
    """Find largest subset with all unordered pair sums unique (including diagonals)."""
    n = len(subset)
    # Try sizes from n down to 1
    for size in range(n, 0, -1):
        if deadline and time.time() > deadline:
            return 0
        for combo in combinations(range(n), size):
            if deadline and time.time() > deadline:
                return 0
            vals = [subset[i] for i in combo]
            if _is_sidon(vals):
                return size
    return 1

def _local_search(candidate, deadline, max_iter=200):
    """
    Local search to maximize duplicate sums (minimize Sidon subset size).
    Uses incremental updates for speed.
    """
    candidate = list(candidate)
    n = len(candidate)
    
    def dup_count(arr):
        sums = [arr[i] + arr[j] for i in range(n) for j in range(i, n)]
        c = Counter(sums)
        return sum(k * (k - 1) // 2 for k in c.values())
    
    current_dups = dup_count(candidate)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try replacing each element with all possible values
        for i in range(n):
            old_val = candidate[i]
            # Try a subset of values for speed
            # Focus on values that create collisions
            for new_val in range(129):
                if new_val == old_val:
                    continue
                # Check if new_val is already in candidate (excluding position i)
                if any(candidate[j] == new_val for j in range(n) if j != i):
                    continue
                
                candidate[i] = new_val
                new_dups = dup_count(candidate)
                
                if new_dups > current_dups:
                    current_dups = new_dups
                    improved = True
                    # Don't break - keep trying to find even better
                else:
                    candidate[i] = old_val
        
        if not improved:
            break
    
    return candidate, current_dups

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Initial candidate: structured set with many collisions
    # Use pairs that sum to the same value
    best_subset = list(range(12))
    best_score = _max_sidon_subset_size(best_subset, deadline)
    
    random.seed(42)
    
    # Generate diverse initial candidates
    initial_candidates = [
        list(range(12)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # Even numbers
        [0, 1, 3, 7, 15, 31, 63, 127, 62, 30, 14, 6],  # Mixed
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 63, 31, 15],  # Powers of 2 + others
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 128],  # One outlier
    ]
    
    for candidate in initial_candidates:
        if time.time() > deadline - 10:
            break
        score = _max_sidon_subset_size(candidate, deadline)
        if score < best_score:
            best_score = score
            best_subset = candidate
            if best_score <= 3:
                break
    
    # Local search with multiple restarts
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        
        # Generate random candidate or perturb best
        if restart % 3 == 0:
            candidate = random.sample(range(129), 12)
        else:
            candidate = list(best_subset)
            # Perturb 1-3 elements
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, 11)
                candidate[idx] = random.randint(0, 128)
                # Ensure uniqueness
                attempts = 0
                while any(candidate[j] == candidate[idx] for j in range(12) if j != idx):
                    candidate[idx] = random.randint(0, 128)
                    attempts += 1
                    if attempts > 20:
                        break
        
        # Local search
        improved_candidate, _ = _local_search(candidate, deadline - 3)
        
        # Evaluate
        if time.time() > deadline - 2:
            break
        
        score = _max_sidon_subset_size(improved_candidate, deadline)
        
        if score < best_score:
            best_score = score
            best_subset = improved_candidate
            if best_score <= 3:
                break
    
    return sorted(best_subset)
# EVOLVE-BLOCK-END
