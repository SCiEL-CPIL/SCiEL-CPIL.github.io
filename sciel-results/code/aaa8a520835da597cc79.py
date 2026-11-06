"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    def is_sidon_subset(subset):
        """Check if a subset is Sidon (all unordered pair sums including diagonals are unique)."""
        sums = set()
        n = len(subset)
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def max_sidon_size(candidate):
        """Find the maximum size of a Sidon subset of candidate."""
        n = len(candidate)
        # Try from largest possible downward
        for size in range(n, 0, -1):
            for subset in combinations(candidate, size):
                if is_sidon_subset(subset):
                    return size
        return 0
    
    def count_sidon_subsets(candidate, max_size=6):
        """Count how many Sidon subsets of size max_size exist (for scoring)."""
        count = 0
        for subset in combinations(candidate, max_size):
            if is_sidon_subset(subset):
                count += 1
        return count
    
    best_set = None
    best_ms = 12  # worst case
    
    # Strategy: search in small ranges to force collisions
    # Try ranges [0, R] for various R
    random.seed(42)
    
    # First, try some structured approaches
    # For small ranges, we want elements close together to maximize collisions
    
    ranges_to_try = [
        (0, 10), (0, 12), (0, 15), (0, 18), (0, 20), (0, 25),
        (0, 30), (0, 40), (5, 35), (10, 40)
    ]
    
    for lo, hi in ranges_to_try:
        if time.time() > deadline:
            break
        range_size = hi - lo + 1
        if range_size < 12:
            continue
        
        # Try random sets in this range
        num_trials = 200
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Generate random 12-element set
            candidate = sorted(random.sample(range(lo, hi + 1), 12))
            
            # Quick check: if max_sidon_size is already good, keep it
            ms = max_sidon_size(candidate)
            if ms < best_ms:
                best_ms = ms
                best_set = candidate
                if best_ms <= 4:
                    return best_set
        
        # Local search from best found in this range
        if best_set is not None and best_ms > 4:
            # Try to improve by swapping elements
            current = list(best_set)
            for iteration in range(50):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(12):
                    for new_val in range(lo, hi + 1):
                        if new_val in current:
                            continue
                        old_val = current[i]
                        current[i] = new_val
                        current.sort()
                        ms = max_sidon_size(current)
                        if ms < best_ms:
                            best_ms = ms
                            best_set = current
                            if best_ms <= 4:
                                return best_set
                        elif ms == best_ms:
                            # Keep this swap if it doesn't make things worse
                            # and try to further reduce collisions
                            pass
                        else:
                            current[i] = old_val
                        current.sort()
                current = list(best_set)
    
    if best_set is None:
        best_set = list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
