"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    def is_sidon(subset):
        """Check if subset has all pair sums unique including diagonals."""
        n = len(subset)
        sums = set()
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def max_sidon_subset_size(ambient):
        """Find the size of the largest Sidon subset of ambient."""
        n = len(ambient)
        # Try from largest to smallest
        for size in range(n, 0, -1):
            for subset in combinations(ambient, size):
                if is_sidon(subset):
                    return size
        return 0
    
    # Start with a promising initial candidate
    # Sets with many additive collisions tend to have smaller max Sidon subsets
    best_set = list(range(12))
    best_size = max_sidon_subset_size(best_set)
    
    # Try some structured candidates first
    candidates = []
    
    # Arithmetic progressions with small step
    candidates.append(list(range(0, 24, 2)))  # 0,2,4,...,22
    candidates.append(list(range(0, 24, 2)))
    
    # Sets with dense additive structure
    candidates.append([0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    candidates.append([0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    candidates.append([0, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
    
    # Dense low-range sets
    candidates.append(list(range(0, 12)))
    candidates.append(list(range(1, 13)))
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Sets with many repeated differences
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21])
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        sz = max_sidon_subset_size(cand)
        if sz < best_size:
            best_size = sz
            best_set = cand[:]
    
    # Local search: mutate the best set
    random.seed(42)
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Create a mutated version
        new_set = best_set[:]
        idx = random.randint(0, 11)
        old_val = new_set[idx]
        new_val = random.randint(0, 128)
        if new_val in new_set and new_val != old_val:
            continue
        new_set[idx] = new_val
        
        sz = max_sidon_subset_size(new_set)
        if sz < best_size:
            best_size = sz
            best_set = new_set[:]
    
    return best_set
# EVOLVE-BLOCK-END
