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
    2. Local search with simulated annealing (fast conflict tracking)
    3. Multiple restarts with different random seeds
    4. Final high-precision polish pass
    5. Deadline guard to ensure we finish within 600s
    """
    m = 205
    k = 2
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 595.0  # 5s safety margin
    
    # Forbidden differences: the nonzero k-th power residues modulo m.
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # Precompute forbidden differences as a set for O(1) lookup
    forbidden_set = set(forbidden)
    forbidden_list = list(forbidden_set)
    
    # Precompute conflict mask: for each element, which elements conflict with it
    # conflict_mask[i] is a bitmask of elements that conflict with element i
    conflict_mask = np.zeros(m, dtype=np.int64)
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
        selected_set = set()
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
                selected_set.add(candidate)
        return selected
    
    def local_search(initial_subset, max_iterations=20000):
        """
        Local search using simulated annealing with fast conflict tracking.
        Tries to add elements, remove elements, or swap elements.
        """
        current = list(initial_subset)
        current_set = set(current)
        current_size = len(current)
        best = list(current)
        best_size = current_size
        
        # Precompute conflicts for the current set
        # conflict_count[x] = number of elements in current that conflict with x
        conflict_count = np.zeros(m, dtype=np.int32)
        for s in current:
            mask = conflict_mask[s]
            # Add conflicts
            for x in range(m):
                if (mask >> x) & 1:
                    conflict_count[x] += 1
        
        def get_conflicts(subset):
            """Return conflict counts for the subset."""
            cc = np.zeros(m, dtype=np.int32)
            for s in subset:
                mask = conflict_mask[s]
                for x in range(m):
                    if (mask >> x) & 1:
                        cc[x] += 1
            return cc
        
        temperature = 10.0
        cooling_rate = 0.9998
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
                # Find available elements (not in current, not conflicting)
                available = []
                for x in range(m):
                    if x not in current_set and conflict_count[x] == 0:
                        available.append(x)
                if available:
                    candidate = random.choice(available)
                    new_subset = current + [candidate]
                    new_set = current_set | {candidate}
                    new_conflict_count = conflict_count.copy()
                    mask = conflict_mask[candidate]
                    for x in range(m):
                        if (mask >> x) & 1:
                            new_conflict_count[x] += 1
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
                            current_set = new_set
                            conflict_count = new_conflict_count
                            current_size += 1
                            if current_size > best_size:
                                best = list(current)
                                best_size = current_size
            elif move_type < 0.7:
                # Try to remove an element
                if current:
                    candidate = random.choice(current)
                    new_subset = [x for x in current if x != candidate]
                    new_set = current_set - {candidate}
                    new_conflict_count = conflict_count.copy()
                    mask = conflict_mask[candidate]
                    for x in range(m):
                        if (mask >> x) & 1:
                            new_conflict_count[x] -= 1
                    delta = -1
                    if delta > 0 or random.random() < np.exp(delta / temperature):
                        current = new_subset
                        current_set = new_set
                        conflict_count = new_conflict_count
                        current_size -= 1
            else:
                # Try to swap: remove one, add another
                if current:
                    to_remove = random.choice(current)
                    # Find available elements to add
                    available = []
                    for x in range(m):
                        if x not in current_set and x != to_remove:
                            # Check if x would be valid after removing to_remove
                            # x must not conflict with any element in current except possibly to_remove
                            mask = conflict_mask[x]
                            valid = True
                            for s in current:
                                if s == to_remove:
                                    continue
                                if (mask >> s) & 1:
                                    valid = False
                                    break
                            if valid:
                                available.append(x)
                    if available:
                        to_add = random.choice(available)
                        new_subset = [x for x in current if x != to_remove] + [to_add]
                        new_set = (current_set - {to_remove}) | {to_add}
                        new_conflict_count = conflict_count.copy()
                        # Remove conflicts from to_remove
                        mask_rm = conflict_mask[to_remove]
                        for x in range(m):
                            if (mask_rm >> x) & 1:
                                new_conflict_count[x] -= 1
                        # Add conflicts from to_add
                        mask_add = conflict_mask[to_add]
                        for x in range(m):
                            if (mask_add >> x) & 1:
                                new_conflict_count[x] += 1
                        delta = 0
                        if random.random() < np.exp(delta / temperature):
                            current = new_subset
                            current_set = new_set
                            conflict_count = new_conflict_count
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
    num_restarts = 80
    
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
        result = local_search(initial, max_iterations=10000)
        
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
    
    # Final high-precision polish pass: try swaps to improve
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        improved = False
        for i, a in enumerate(best_overall):
            for b in range(m):
                if b in best_overall:
                    continue
                # Try swapping a with b
                new_set = [x for x in best_overall if x != a]
                can_add = True
                for s in new_set:
                    if not is_valid_pair(b, s):
                        can_add = False
                        break
                if can_add:
                    new_set.append(b)
                    if len(new_set) > len(best_overall):
                        best_overall = new_set
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    # Verify the solution is valid
    assert is_valid_subset(best_overall), "Solution is not valid!"
    assert len(set(best_overall)) == len(best_overall), "Duplicates found!"
    assert all(0 <= x <= m - 1 for x in best_overall), "Out of range!"
    
    return np.array(sorted(best_overall), dtype=np.int64)


# EVOLVE-BLOCK-END
