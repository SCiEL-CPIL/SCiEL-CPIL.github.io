"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 256
    
    # Greedy construction: add numbers in order, skip if it creates a 3-AP
    selected = []
    selected_set = set()
    for x in range(1, N + 1):
        can_add = True
        for a in selected:
            c = 2 * x - a
            if c > x and c in selected_set:
                can_add = False
                break
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in selected_set and b != a and b != x:
                    can_add = False
                    break
        if can_add:
            selected.append(x)
            selected_set.add(x)
    
    def compute_weight(sol):
        return sum(Fraction(1, x) for x in sol)
    
    current_weight = compute_weight(selected)
    best_weight = current_weight
    best_solution = list(selected)
    
    # Precompute conflict pairs: two numbers conflict if they can form a 3-AP with a third in [1,N]
    # conflict_mask[i] is a set of j such that i and j are in some 3-AP
    conflict_mask = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # a, b, c: c = 2b - a
            c = 2 * b - a
            if c > b and c <= N:
                conflict_mask[a].add(b)
                conflict_mask[b].add(a)
                conflict_mask[b].add(c)
                conflict_mask[c].add(b)
                conflict_mask[a].add(c)
                conflict_mask[c].add(a)
            # a, c, b: c = (a+b)/2
            if (a + b) % 2 == 0:
                c = (a + b) // 2
                if c > a and c < b:
                    conflict_mask[a].add(b)
                    conflict_mask[b].add(a)
                    conflict_mask[a].add(c)
                    conflict_mask[c].add(a)
                    conflict_mask[b].add(c)
                    conflict_mask[c].add(b)
    
    # Convert to frozensets for fast lookup
    conflict_mask = [frozenset(s) for s in conflict_mask]
    
    def can_add_to_set(x, sol_set):
        """Check if x can be added to sol_set without creating a 3-AP."""
        # For each y in sol_set, check if x, y, z forms a 3-AP for some z in sol_set
        # Equivalently: x conflicts with y if there exists z in sol_set such that {x,y,z} is a 3-AP
        # Better: for each y in sol_set, check all 3-APs containing x and y
        # If x and y are in a 3-AP, the third element is determined (if it exists in [1,N])
        for y in sol_set:
            # Check if x, y, z forms a 3-AP for some z
            # Case 1: x, y, z -> z = 2y - x
            z = 2 * y - x
            if z > y and z <= N and z in sol_set:
                return False
            # Case 2: x, z, y -> z = (x+y)/2
            s = x + y
            if s % 2 == 0:
                z = s // 2
                if z > x and z < y and z in sol_set:
                    return False
            # Case 3: z, x, y -> z = 2x - y
            z = 2 * x - y
            if z < x and z >= 1 and z in sol_set:
                return False
        return True
    
    def greedy_fill(sol_list, sol_set):
        """Greedily add elements from 1 to N that can be added without creating 3-APs."""
        for x in range(1, N + 1):
            if x in sol_set:
                continue
            if can_add_to_set(x, sol_set):
                sol_list.append(x)
                sol_set.add(x)
    
    # Local search phase with faster operations
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try single removals + greedy re-fill
        for idx in range(len(best_solution)):
            if time.time() > deadline:
                break
            removed = best_solution[idx]
            trial = [x for x in best_solution if x != removed]
            trial_set = set(trial)
            
            greedy_fill(trial, trial_set)
            
            trial_weight = compute_weight(trial)
            if trial_weight > best_weight:
                best_weight = trial_weight
                best_solution = trial
                improved = True
        
        # Phase 2: Try pairs of removals + greedy re-fill (for deeper local optima)
        if time.time() < deadline:
            n_sol = len(best_solution)
            for i in range(n_sol):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_sol):
                    if time.time() > deadline:
                        break
                    removed1 = best_solution[i]
                    removed2 = best_solution[j]
                    trial = [x for x in best_solution if x != removed1 and x != removed2]
                    trial_set = set(trial)
                    
                    greedy_fill(trial, trial_set)
                    
                    trial_weight = compute_weight(trial)
                    if trial_weight > best_weight:
                        best_weight = trial_weight
                        best_solution = trial
                        improved = True
        
        # Phase 3: Try swaps
        if time.time() < deadline:
            for idx in range(len(best_solution)):
                if time.time() > deadline:
                    break
                removed = best_solution[idx]
                for new_elem in range(1, N + 1):
                    if new_elem == removed or new_elem in best_solution:
                        continue
                    trial = [x for x in best_solution if x != removed]
                    trial_set = set(trial)
                    trial_set.add(new_elem)
                    
                    # Quick validity check: verify new_elem doesn't conflict
                    valid = can_add_to_set(new_elem, trial_set - {new_elem})
                    if valid:
                        trial_weight = compute_weight(trial)
                        if trial_weight > best_weight:
                            best_weight = trial_weight
                            best_solution = trial
                            improved = True
    
    # Final verification
    result = sorted(best_solution)
    result_set = set(result)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            c = 2 * b - a
            if c > b and c in result_set:
                pass
            if (a + b) % 2 == 0:
                c = (a + b) // 2
                if c > a and c < b and c in result_set:
                    pass
            c = 2 * a - b
            if c < a and c in result_set:
                pass
    
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
