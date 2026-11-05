"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _is_sidon(subset):
    """Check if subset has all unordered pair sums unique including diagonals."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(arr):
    """Exhaustively find the largest Sidon subset of arr."""
    n = len(arr)
    best = 0
    # Try all subsets from largest to smallest
    for size in range(n, 0, -1):
        for subset in itertools.combinations(arr, size):
            if _is_sidon(subset):
                return size
    return 0

def _collision_count(arr):
    """Count the number of additive collisions (pairs of pairs with same sum).
    Higher collision count means harder to extract large Sidon subsets."""
    n = len(arr)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sums[s] = sums.get(s, 0) + 1
    # Total number of colliding pairs of pairs
    collisions = 0
    for count in sums.values():
        if count > 1:
            collisions += count * (count - 1) // 2
    return collisions

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
    best_arr = list(range(12))
    best_score = _max_sidon_size(best_arr)
    
    # Use simulated annealing to maximize collision count
    # which correlates with smaller max Sidon subset size
    
    arr = list(range(12))
    current_collisions = _collision_count(arr)
    current_sidon = best_score
    
    T = 10.0
    cooling = 0.9995
    rng = random.Random(42)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Random move: pick a random element and change its value
        idx = rng.randint(0, 11)
        old_val = arr[idx]
        new_val = rng.randint(0, 128)
        
        # Skip if same value
        if new_val == old_val:
            continue
        
        # Check uniqueness
        if new_val in arr:
            continue
        
        arr[idx] = new_val
        new_collisions = _collision_count(arr)
        
        # Acceptance criterion (simulated annealing)
        delta = new_collisions - current_collisions
        if delta > 0 or rng.random() < pow(2.718281828, delta / T):
            current_collisions = new_collisions
            # Occasionally evaluate actual Sidon size (expensive)
            if iteration % 50 == 0:
                sidon_size = _max_sidon_size(arr)
                if sidon_size < current_sidon:
                    current_sidon = sidon_size
                    best_arr = arr[:]
                    best_score = sidon_size
                    if best_score <= 4:
                        break
        else:
            arr[idx] = old_val
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    # Final check
    final_sidon = _max_sidon_size(best_arr)
    if final_sidon < best_score:
        best_score = final_sidon
    
    return best_arr
# EVOLVE-BLOCK-END
