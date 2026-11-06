"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if subset has all pair sums (including diagonals) unique."""
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
    """Find the maximum size of a Sidon subset of arr."""
    n = len(arr)
    # Try from largest to smallest
    for k in range(n, 0, -1):
        for combo in combinations(range(n), k):
            subset = [arr[i] for i in combo]
            if _is_sidon(subset):
                return k
    return 0

def _collision_score(arr):
    """Count sum collisions - higher means more collisions (harder to find Sidon sets)."""
    n = len(arr)
    sum_counts = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
    # Score = number of sums that appear more than once (weighted)
    score = 0
    for s, c in sum_counts.items():
        if c > 1:
            score += c * (c - 1) // 2  # number of colliding pairs
    return score

def construct():
    deadline = time.time() + 80  # safety margin
    
    # Initial random set
    best_set = sorted(random.sample(range(129), 12))
    best_sidon_size = _max_sidon_size(best_set)
    
    # Simulated annealing
    current_set = list(best_set)
    current_sidon = best_sidon_size
    current_collision = _collision_score(current_set)
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        T *= cooling
        if T < 0.01:
            break
        
        # Pick a random element to perturb
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Try a new value
        new_val = random.randint(0, 128)
        if new_val in current_set:
            continue
        
        current_set[idx] = new_val
        new_collision = _collision_score(current_set)
        
        # Compute new Sidon size (expensive, so only do it if collision improved or randomly)
        delta_collision = new_collision - current_collision
        
        if delta_collision > 0 or random.random() < T:
            new_sidon = _max_sidon_size(current_set)
            if new_sidon < best_sidon_size:
                best_sidon_size = new_sidon
                best_set = list(current_set)
            current_sidon = new_sidon
            current_collision = new_collision
        else:
            # Reject
            current_set[idx] = old_val
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
