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
    deadline = 85.0  # Safety margin before 90s kill
    
    random.seed(42)
    
    # Start with sets in small range to force collisions
    best_set = None
    best_score = 12  # worst case
    
    # Try many random sets in small ranges
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        max_range = random.choice([20, 25, 30, 40, 50, 60, 80, 128])
        s = random.sample(range(max_range + 1), 12)
        score = _largest_sidon_size(s)
        if score < best_score:
            best_score = score
            best_set = s[:]
        if best_score <= 4:
            break
    
    if best_set is None:
        best_set = list(range(12))
        best_score = _largest_sidon_size(best_set)
    
    current = best_set[:]
    current_score = best_score
    
    # Local search with swaps
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        improved = False
        # Try random swaps
        for _ in range(3000):
            if time.time() >= deadline - 5:
                break
            i = random.randint(0, 11)
            # Try replacing current[i] with a random value not in current
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
    
    # Final polish: try to improve best_set with targeted moves
    if time.time() < deadline - 3:
        for _ in range(10000):
            if time.time() >= deadline - 2:
                break
            candidate = best_set[:]
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in candidate:
                continue
            candidate[i] = new_val
            score = _largest_sidon_size(candidate)
            if score < best_score:
                best_score = score
                best_set = candidate[:]
    
    # Verify best_set is valid
    assert len(set(best_set)) == 12
    assert all(0 <= x <= 128 for x in best_set)
    
    return best_set
# EVOLVE-BLOCK-END
