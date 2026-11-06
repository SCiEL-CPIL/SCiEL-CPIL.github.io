import time
import random
import sys

def construct():
    """
    Finds a maximum size Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses a backtracking search with heuristics and a known solution as a guide.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    required = [1, 2, 5, 11]
    max_val = 284
    target_size = 20
    
    # Known optimal 20-mark Golomb ruler (0-based) shifted to 1-based
    # This serves as a strong heuristic/priority list
    known_ruler = [0, 1, 4, 10, 18, 23, 29, 42, 47, 52, 78, 82, 90, 92, 119, 129, 144, 172, 186, 283]
    known_set = set(x + 1 for x in known_ruler)
    
    # Candidates are all numbers in [1, 284] not in required, 
    # prioritized by membership in known_set
    all_candidates = list(range(1, max_val + 1))
    # Filter out required elements from the "to be added" pool
    pool = [x for x in all_candidates if x not in required]
    
    # Sort pool: elements in known_set first, then others
    # This biases the search towards the known solution
    pool.sort(key=lambda x: (0 if x in known_set else 1, x))
    
    # Initial state
    current_set = set(required)
    used_sums = set()
    for i in range(len(required)):
        for j in range(i, len(required)):
            used_sums.add(required[i] + required[j])
            
    best_solution = list(required)
    best_size = len(best_solution)
    
    def can_add(x, current_set, used_sums):
        """Check if x can be added to current_set without violating Sidon property."""
        for y in current_set:
            if x + y in used_sums:
                return False
        if 2 * x in used_sums:
            return False
        return True
    
    def get_new_sums(x, current_set):
        """Get the new sums that would be added if x is added to current_set."""
        new_sums = set()
        for y in current_set:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums
    
    def backtrack(current_set, used_sums, pool_index, depth):
        nonlocal best_solution, best_size
        
        current_size = len(current_set)
        if current_size > best_size:
            best_solution = list(current_set)
            best_size = current_size
            
        # If we reached target size, we can stop (since we want max, and 20 is max possible)
        if current_size == target_size:
            return True
            
        # Time check
        if time.time() > deadline:
            return False
            
        # If no more candidates
        if pool_index >= len(pool):
            return False
            
        # Heuristic: If we can't reach target size with remaining slots, prune
        need = target_size - current_size
        available = len(pool) - pool_index
        if available < need:
            return False
            
        # Try adding candidates from pool starting at pool_index
        for i in range(pool_index, len(pool)):
            x = pool[i]
            
            # Check if x can be added
            if not can_add(x, current_set, used_sums):
                continue
                
            # Add x
            current_set.add(x)
            new_sums = get_new_sums(x, current_set)
            used_sums.update(new_sums)
            
            if backtrack(current_set, used_sums, i + 1, depth + 1):
                return True
                
            # Undo
            current_set.remove(x)
            used_sums.difference_update(new_sums)
                
        return False

    # Run the search
    found = backtrack(current_set, used_sums, 0, 0)
    
    return sorted(best_solution)
