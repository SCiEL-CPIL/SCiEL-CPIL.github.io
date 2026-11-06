import random
import time
from itertools import combinations

def is_sidon(subset):
    """Check if a subset is a Sidon set (all pair sums unique)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_sidon_subset(S, k):
    """Find a Sidon subset of size k in S. Returns None if not found."""
    for comb in combinations(S, k):
        if is_sidon(list(comb)):
            return list(comb)
    return None

def has_sidon_subset_of_size_k(S, k):
    """Check if S contains any Sidon subset of size k."""
    return find_sidon_subset(S, k) is not None

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Initial guess: Arithmetic Progression (densest collisions)
    best_set = list(range(12))
    
    # Check if best_set has a Sidon subset of size 4
    # If it doesn't, its max size is 3 (since min is 3).
    if not has_sidon_subset_of_size_k(best_set, 4):
        return best_set
        
    # Local Search: Try to break all 4-Sidon subsets
    current_set = best_set[:]
    current_set.sort()
    
    # We will try to modify elements to destroy the Sidon property of size-4 subsets.
    # Strategy: Find a 4-Sidon subset, change one of its elements to create a collision.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Find a 4-Sidon subset in current_set
        sidon_sub = find_sidon_subset(current_set, 4)
        if sidon_sub is None:
            # No 4-Sidon subset found. Max size is 3.
            return current_set
        
        # Pick an element from the Sidon subset to modify
        # We want to create a collision.
        # If we change x to y, we need y such that sums collide.
        # Simple heuristic: Move x slightly, or swap with another number.
        
        # Try random perturbations of the set
        # Pick an index to change
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Try to find a new value that breaks the specific Sidon subset
        # Or just random restart with bias towards dense sets
        
        # Bias: Keep most elements, change one
        # Ensure distinctness and range [0, 128]
        
        # Try values close to old_val or random
        found_breaker = False
        for _ in range(100):
            if random.random() < 0.5:
                new_val = max(0, min(128, old_val + random.randint(-5, 5)))
            else:
                new_val = random.randint(0, 128)
            
            if new_val in current_set and new_val != old_val:
                continue
                
            test_set = current_set[:]
            test_set[idx] = new_val
            test_set.sort()
            
            # Check if this change broke the specific Sidon subset
            # Note: The subset elements might have changed indices, but values are what matter.
            # We just check if the NEW set has ANY 4-Sidon subset.
            # If it has none, we win.
            # If it has one, we continue.
            
            # Optimization: Check if the specific sidon_sub is still Sidon in the new context?
            # No, we need to check if ANY 4-subset is Sidon.
            
            if not has_sidon_subset_of_size_k(test_set, 4):
                return test_set
                
            # If not fully broken, keep it if it seems "denser" or just discard?
            # For now, discard and try another.
            
    return best_set
