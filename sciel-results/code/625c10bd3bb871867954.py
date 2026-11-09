# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/205Z of maximum
    size that does NOT contain any two elements differing by a nonzero perfect square modulo 205.
    
    Uses a combination of:
    1. Deterministic greedy initialization
    2. Local search with simulated annealing
    3. Multiple restarts with different random seeds
    4. Deadline guard to ensure we finish within 630s
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
    
    def is_valid_pair(a, b):
        """Check if two elements form a valid pair (no forbidden difference)."""
        d1 = (a - b) % m
        d2 = (b - a) % m
        return d1 not in forbidden_set and d2 not in forbidden_set
    
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
                d1 = (candidate - s) % m
                d2 = (s - candidate) % m
                if d1 in forbidden_set or d2 in forbidden_set:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected
    
    def local_search(initial_subset, max_iterations=10000):
        """
        Local search using simulated annealing.
        Tries to add elements, remove elements, or swap elements.
        """
        current = list(initial_subset)
        current_size = len(current)
        best = list(current)
        best_size = current_size
        
        # Precompute conflicts for the current set
        def get_conflicts(subset):
            """Return set of elements that conflict with the subset."""
            conflicts = set()
            for s in subset:
                for d in forbidden_set:
                    conflicts.add((s + d) % m)
                    conflicts.add((s - d) % m)
            return conflicts
        
        current_conflicts = get_conflicts(current)
        
        temperature = 10.0
        cooling_rate = 0.9995
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
                available = [x for x in range(m) if x not in current and x not in current_conflicts]
                if available:
                    candidate = random.choice(available)
                    new_subset = current + [candidate]
                    new_conflicts = current_conflicts | get_conflicts([candidate])
                    # Check if adding creates internal conflicts
                    valid_add = True
                    for s in current:
                        if not is_valid_pair(candidate, s):
                            valid_add = False
                            break
                    if valid_add:
                        delta = 1
                        if delta > 0 or random.random() < np.exp(delta / temperature):
                            current = new_subset
                            current_conflicts = new_conflicts
                            current_size += 1
                            if current_size > best_size:
                                best = list(current)
                                best_size = current_size
            elif move_type < 0.7:
                # Try to remove an element
                if current:
                    candidate = random.choice(current)
                    new_subset = [x for x in current if x != candidate]
                    new_conflicts = get_conflicts(new_subset)
                    delta = -1
                    if delta > 0 or random.random() < np.exp(delta / temperature):
                        current = new_subset
                        current_conflicts = new_conflicts
                        current_size -= 1
            else:
                # Try to swap: remove one, add another
                if current:
                    to_remove = random.choice(current)
                    available = [x for x in range(m) if x not in current and x != to_remove]
                    if available:
                        to_add = random.choice(available)
                        new_subset = [x for x in current if x != to_remove] + [to_add]
                        # Check validity
                        valid = True
                        for s in new_subset:
                            if s == to_add:
                                continue
                            if not is_valid_pair(to_add, s):
                                valid = False
                                break
                        if valid:
                            new_conflicts = get_conflicts(new_subset)
                            delta = 0
                            if random.random() < np.exp(delta / temperature):
                                current = new_subset
                                current_conflicts = new_conflicts
                                if current_size > best_size:
                                    best = list(current)
                                    best_size = current_size
            
            temperature *= cooling_rate
            iteration += 1
        
        return best
    
    # Start with greedy solution
    best_overall = greedy_construct()
    best_size_overall = len(best_overall)
    
    # Try multiple restarts with different random seeds
    num_restarts = 50
    
    for restart in range(num_restarts):
        # Check deadline
        if time.time() > deadline - 60:
            break
        
        # Create a perturbed initial solution
        initial = list(best_overall)
        
        # Randomly remove some elements
        num_remove = random.randint(0, min(3, len(initial)))
        if num_remove > 0 and len(initial) > num_remove:
            to_remove = random.sample(initial, num_remove)
            initial = [x for x in initial if x not in to_remove]
        
        # Randomly add some elements
        num_add = random.randint(0, 2)
        for _ in range(num_add):
            available = [x for x in range(m) if x not in initial]
            if available:
                candidate = random.choice(available)
                # Check if it can be added
                can_add = True
                for s in initial:
                    if not is_valid_pair(candidate, s):
                        can_add = False
                        break
                if can_add:
                    initial.append(candidate)
        
        # Run local search
        result = local_search(initial, max_iterations=5000)
        
        if len(result) > best_size_overall:
            best_overall = result
            best_size_overall = len(result)
        
        # Check deadline
        if time.time() > deadline - 30:
            break
    
    # Final polish: try to add any remaining elements
    for candidate in range(m):
        if candidate not in best_overall:
            can_add = True
            for s in best_overall:
                if not is_valid_pair(candidate, s):
                    can_add = False
                    break
            if can_add:
                best_overall.append(candidate)
    
    # Verify the solution is valid
    assert is_valid_subset(best_overall), "Solution is not valid!"
    assert len(set(best_overall)) == len(best_overall), "Duplicates found!"
    assert all(0 <= x <= m - 1 for x in best_overall), "Out of range!"
    
    return np.array(sorted(best_overall), dtype=np.int64)


# EVOLVE-BLOCK-END
