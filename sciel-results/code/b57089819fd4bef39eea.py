"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if a subset is Sidon (all pair sums including diagonals are unique)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _largest_sidon_size(arr):
    """Find the size of the largest Sidon subset of arr (12 elements)."""
    n = len(arr)
    # Try from largest possible down
    for k in range(n, 1, -1):
        for combo in combinations(arr, k):
            if _is_sidon(combo):
                return k
    return 1  # Any single element is Sidon

def construct():
    """Find 12 distinct integers in [0,128] minimizing the largest Sidon subset size."""
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    best_set = list(range(12))
    best_score = _largest_sidon_size(best_set)
    
    random.seed(42)
    
    # Initialize with a random set
    current = random.sample(range(129), 12)
    current_score = _largest_sidon_size(current)
    if current_score < best_score:
        best_set = current[:]
        best_score = current_score
    
    # Local search with swaps
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        # Try random swaps
        for _ in range(1000):
            if time.time() >= deadline:
                break
            i, j = random.sample(range(12), 2)
            # Try swapping current[i] with a random value not in current
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            
            old_val = current[i]
            current[i] = new_val
            new_score = _largest_sidon_size(current)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_set = current[:]
                    no_improve_count = 0
            else:
                # Revert
                current[i] = old_val
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
        
        # If we've found a good score, do more targeted search
        if best_score <= 4:
            break
    
    # Verify best_set is valid
    assert len(set(best_set)) == 12
    assert all(0 <= x <= 128 for x in best_set)
    
    return best_set
# EVOLVE-BLOCK-END
