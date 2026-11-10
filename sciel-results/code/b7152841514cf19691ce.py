# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/205Z of maximum
    size that does NOT contain any two elements differing by a nonzero perfect square modulo 205.
    
    Uses bitset-based conflict tracking for fast conflict checks,
    systematic 1-1 swap local search, and multiple restarts.
    """
    m = 205
    k = 2
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0  # 600s safety margin
    
    # Forbidden differences: the nonzero k-th power residues modulo m.
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # Precompute forbidden differences as a set for O(1) lookup
    forbidden_set = set(forbidden)
    
    # Precompute conflict bitsets: for each element, which other elements conflict with it
    # conflict_mask[i] is a bitmask where bit j is set if i and j have a forbidden difference
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden_set:
            j1 = (i + d) % m
            j2 = (i - d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        conflict_mask[i] = mask
    
    def is_valid_pair(a, b):
        """Check if two elements form a valid pair (no forbidden difference)."""
        return (conflict_mask[a] >> b) & 1 == 0
    
    def is_valid_subset(subset):
        """Check if the entire subset is valid."""
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                if not is_valid_pair(subset[i], subset[j]):
                    return False
        return True
    
    def greedy_construct():
        """Deterministic greedy construction."""
        selected = []
        for candidate in range(m):
            ok = True
            for s in selected:
                if not is_valid_pair(candidate, s):
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected
    
    def get_conflict_bitset(subset):
        """Return the union of conflict masks for all elements in subset."""
        mask = 0
        for s in subset:
            mask |= conflict_mask[s]
        return mask
    
    def get_available_elements(current_set, current_conflict_mask):
        """Get elements not in current_set and not conflicting with it."""
        available = []
        for x in range(m):
            if x not in current_set and not (current_conflict_mask >> x) & 1:
                available.append(x)
        return available
    
    def local_search(initial_subset, max_iterations=20000):
        """
        Local search using simulated annealing with bitset-based conflict tracking.
        Tries to add elements, remove elements, or swap elements.
        """
        current = list(initial_subset)
        current_set = set(current)
        current_size = len(current)
        best = list(current)
        best_size = current_size
        
        # Precompute conflicts for the current set
        current_conflict_mask = get_conflict_bitset(current)
        
        temperature = 10.0
        cooling_rate = 0.99995
        min_temperature = 0.01
        
        iteration = 0
        while temperature > min_temperature and iteration < max_iterations:
            # Check deadline
            if time.time() > deadline - 30:
                break
            
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add an element
                available = get_available_elements(current_set, current_conflict_mask)
                
                if available:
                    candidate = random.choice(available)
                    current.append(candidate)
                    current_set.add(candidate)
                    current_conflict_mask |= conflict_mask[candidate]
                    current_size += 1
                    if current_size > best_size:
                        best = list(current)
                        best_size = current_size
            elif move_type < 0.7:
                # Try to remove an element
                if current:
                    idx = random.randrange(len(current))
                    candidate = current[idx]
                    current.pop(idx)
                    current_set.discard(candidate)
                    current_size -= 1
                    current_conflict_mask = get_conflict_bitset(current)
            else:
                # Try to swap: remove one, add another
                if current:
                    to_remove = random.choice(current)
                    # Compute conflict mask without to_remove
                    temp_conflict = current_conflict_mask
                    # We need to remove to_remove's contribution, but that's tricky with OR
                    # Instead, rebuild from current minus to_remove
                    temp_current = [x for x in current if x != to_remove]
                    temp_conflict = get_conflict_bitset(temp_current)
                    
                    available = get_available_elements(current_set, temp_conflict)
                    # Filter out elements that conflict with temp_current (already done by mask)
                    # But also exclude to_remove itself from "not in current"
                    available = [x for x in available if x != to_remove]
                    
                    if available:
                        to_add = random.choice(available)
                        current.remove(to_remove)
                        current_set.discard(to_remove)
                        current.append(to_add)
                        current_set.add(to_add)
                        current_conflict_mask = get_conflict_bitset(current)
                        if current_size > best_size:
                            best = list(current)
                            best_size = current_size
            
            temperature *= cooling_rate
            iteration += 1
        
        return best
    
    def systematic_swap_search(initial_subset, max_passes=5):
        """
        Systematic 1-1 swap local search: for each element in the subset,
        try all possible replacements. This explores a much larger neighborhood
        than random swaps.
        """
        current = list(initial_subset)
        current_set = set(current)
        current_size = len(current)
        best = list(current)
        best_size = current_size
        
        for _pass in range(max_passes):
            if time.time() > deadline - 30:
                break
            
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 30:
                    break
                
                to_remove = current[i]
                temp_current = [x for x in current if x != to_remove]
                temp_conflict = get_conflict_bitset(temp_current)
                
                # Find all elements that can replace to_remove
                for x in range(m):
                    if x in current_set:
                        continue
                    if (temp_conflict >> x) & 1:
                        continue
                    # The conflict mask already ensures no conflict with temp_current
                    # So this swap is valid
                    current[i] = x
                    current_set.discard(to_remove)
                    current_set.add(x)
                    improved = True
                    break  # Only do one swap per element per pass
            
            if not improved:
                break
        
        return best
    
    # Start with greedy solution
    best_overall = greedy_construct()
    best_set_overall = set(best_overall)
    best_size_overall = len(best_overall)
    
    # Try multiple restarts with different random seeds
    num_restarts = 100
    
    for restart in range(num_restarts):
        # Check deadline
        if time.time() > deadline - 60:
            break
        
        # Create a perturbed initial solution
        initial = list(best_overall)
        initial_set = set(initial)
        
        # Randomly remove some elements
        num_remove = random.randint(0, min(3, len(initial)))
        if num_remove > 0 and len(initial) > num_remove:
            to_remove = random.sample(initial, num_remove)
            to_remove_set = set(to_remove)
            initial = [x for x in initial if x not in to_remove_set]
            initial_set = set(initial)
        
        # Randomly add some elements
        num_add = random.randint(0, 2)
        for _ in range(num_add):
            available = get_available_elements(initial_set, get_conflict_bitset(initial))
            if available:
                candidate = random.choice(available)
                initial.append(candidate)
                initial_set.add(candidate)
        
        # Run local search
        result = local_search(initial, max_iterations=10000)
        
        # Run systematic swap search on the result
        result = systematic_swap_search(result, max_passes=3)
        
        if len(result) > best_size_overall:
            best_overall = result
            best_set_overall = set(result)
            best_size_overall = len(result)
        
        # Check deadline
        if time.time() > deadline - 30:
            break
    
    # Final polish: try to add any remaining elements
    best_conflict = get_conflict_bitset(best_overall)
    for candidate in range(m):
        if candidate not in best_set_overall:
            if not (best_conflict >> candidate) & 1:
                best_overall.append(candidate)
                best_set_overall.add(candidate)
                best_conflict |= conflict_mask[candidate]
    
    # Verify the solution is valid
    assert is_valid_subset(best_overall), "Solution is not valid!"
    assert len(set(best_overall)) == len(best_overall), "Duplicates found!"
    assert all(0 <= x <= m - 1 for x in best_overall), "Out of range!"
    
    return np.array(sorted(best_overall), dtype=np.int64)


# EVOLVE-BLOCK-END
