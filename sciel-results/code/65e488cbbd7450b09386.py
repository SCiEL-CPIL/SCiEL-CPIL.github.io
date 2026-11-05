"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter
from itertools import combinations

def _count_duplicate_sums(subset):
    """Count how many unordered pair sums (including diagonals) are duplicated."""
    n = len(subset)
    sums = [subset[i] + subset[j] for i in range(n) for j in range(i, n)]
    c = Counter(sums)
    return sum(k * (k - 1) // 2 for k in c.values())

def _max_sidon_subset_size_bruteforce(subset, deadline=None):
    """Find largest subset with all unordered pair sums unique (including diagonals)."""
    n = len(subset)
    best = 0
    # Try all subsets in descending size order
    for size in range(n, best, -1):
        if deadline and time.time() > deadline:
            break
        for combo in combinations(range(n), size):
            if deadline and time.time() > deadline:
                break
            vals = [subset[i] for i in combo]
            s = len(vals)
            pair_sums = set()
            is_sidon = True
            for i in range(s):
                for j in range(i, s):
                    ps = vals[i] + vals[j]
                    if ps in pair_sums:
                        is_sidon = False
                        break
                    pair_sums.add(ps)
                if not is_sidon:
                    break
            if is_sidon:
                best = size
                break
        if best == size:
            break
    return best

def _hill_climb(subset, deadline, max_iter=10000):
    """
    Hill climbing to maximize duplicate sums (minimize Sidon subset size).
    Uses incremental updates for speed.
    """
    subset = list(subset)
    n = len(subset)
    
    # Precompute sum counts
    sums = [subset[i] + subset[j] for i in range(n) for j in range(i, n)]
    sum_counts = Counter(sums)
    current_dups = sum(k * (k - 1) // 2 for k in sum_counts.values())
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try replacing each element
        for i in range(n):
            old_val = subset[i]
            # Try all possible new values
            # To speed up, we can iterate through 0..128
            best_new_val = old_val
            best_new_dups = current_dups
            
            for new_val in range(129):
                if new_val == old_val:
                    continue
                if new_val in subset and new_val != old_val:
                    continue
                
                # Calculate delta in duplicates
                # Remove contributions of old_val
                # old_val pairs with j (j != i)
                # old_val pairs with itself (i, i)
                
                # This incremental update is complex. 
                # Given N=12, recalculating _count_duplicate_sums is O(N^2) = 144 ops.
                # 12 * 129 * 144 ~ 225,000 ops per full pass.
                # 1000 passes = 225M ops. Might be too slow in Python.
                
                # Let's stick to the simpler but slightly slower approach for now,
                # but optimize the inner loop.
                
                # Actually, let's just use the random swap from inspiration but with more iterations
                # and a better acceptance criterion (Simulated Annealing).
                pass

    return subset

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Initial candidate: range(12)
    best_subset = list(range(12))
    best_score = _max_sidon_subset_size_bruteforce(best_subset, deadline)
    
    # If we find a score of 3, we can stop early (theoretical min for 12 elements is likely 3 or 4)
    # Min Sidon size for 12 elements:
    # If max Sidon is 3, then any 4 elements have a collision.
    
    random.seed(42)
    
    # Use a more robust local search:
    # 1. Random restarts
    # 2. Hill climbing on duplicate sums
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        # Generate random candidate
        candidate = random.sample(range(129), 12)
        
        # Local search: Maximize duplicates
        current_dups = _count_duplicate_sums(candidate)
        
        for step in range(1000):
            if time.time() > deadline - 2:
                break
            
            # Try a random swap
            i = random.randint(0, 11)
            old_val = candidate[i]
            new_val = random.randint(0, 128)
            
            if new_val in candidate and new_val != old_val:
                continue
                
            candidate[i] = new_val
            new_dups = _count_duplicate_sums(candidate)
            
            if new_dups > current_dups:
                current_dups = new_dups
            else:
                candidate[i] = old_val
        
        # Evaluate the result
        score = _max_sidon_subset_size_bruteforce(candidate, deadline)
        
        if score < best_score:
            best_score = score
            best_subset = candidate
            if best_score <= 3:
                break
                
    return sorted(best_subset)
# EVOLVE-BLOCK-END
