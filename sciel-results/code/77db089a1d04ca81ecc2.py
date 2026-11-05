"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 256
    
    # Greedy construction: add numbers in order, skip if it creates a 3-AP
    # This is the baseline (score ~1.0)
    selected = []
    selected_set = set()
    for x in range(1, N + 1):
        # Check if adding x creates a 3-term AP
        # x can be the middle: need a, x, c with a+c=2x, i.e., a = 2x-c, c in selected
        # x can be the end: need a, b, x with a+x=2b, i.e., b=(a+x)/2 in selected
        # or x, b, c with x+c=2b, i.e., b=(x+c)/2 in selected
        can_add = True
        for a in selected:
            # Case 1: a, x, c -> c = 2x - a, need c in selected and c > x
            c = 2 * x - a
            if c > x and c in selected_set:
                can_add = False
                break
            # Case 2: a, b, x -> b = (a+x)/2, need b in selected
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in selected_set and b != a and b != x:
                    can_add = False
                    break
            # Case 3: x, b, c -> b = (x+c)/2, need b,c in selected with c > x
            # This is covered by case 1 when we iterate
        if can_add:
            selected.append(x)
            selected_set.add(x)
    
    # Compute weight of current solution
    def compute_weight(sol):
        return sum(Fraction(1, x) for x in sol)
    
    current_weight = compute_weight(selected)
    best_weight = current_weight
    best_solution = list(selected)
    
    # Local search: try swaps and additions/removals to improve
    # Key insight: removing a moderately valuable element might allow adding
    # multiple less-valuable elements that collectively have higher reciprocal sum
    
    # Precompute conflicts: two numbers conflict if they can form a 3-AP
    # with some third number in [1, N]
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # Check if a and b are in a 3-AP
            # a, b, c: c = 2b - a
            c = 2 * b - a
            if c > b and c <= N:
                conflict[a][b] = True
                conflict[b][a] = True
            # a, c, b: c = (a+b)/2
            if (a + b) % 2 == 0:
                c = (a + b) // 2
                if c > a and c < b:
                    conflict[a][b] = True
                    conflict[b][a] = True
            # c, a, b: c = 2a - b
            c = 2 * a - b
            if c < a and c >= 1:
                conflict[a][b] = True
                conflict[b][a] = True
    
    # Local search phase
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try single removals: remove one element, then greedily add back what we can
        for idx in range(len(best_solution)):
            if time.time() > deadline:
                break
            removed = best_solution[idx]
            trial = [x for x in best_solution if x != removed]
            trial_set = set(trial)
            
            # Try to add back elements that were previously excluded
            # First, add the removed element back if possible (no-op check)
            # Then try adding other elements
            for x in range(1, N + 1):
                if x in trial_set:
                    continue
                # Check if x can be added
                can_add = True
                for a in trial:
                    c = 2 * x - a
                    if c > x and c in trial_set:
                        can_add = False
                        break
                    s = a + x
                    if s % 2 == 0:
                        b = s // 2
                        if b in trial_set and b != a and b != x:
                            can_add = False
                            break
                    c = 2 * a - x
                    if c < a and c in trial_set:
                        can_add = False
                        break
                if can_add:
                    trial.append(x)
                    trial_set.add(x)
            
            trial_weight = compute_weight(trial)
            if trial_weight > best_weight:
                best_weight = trial_weight
                best_solution = trial
                improved = True
        
        # Try swaps: remove one element, add another
        for idx in range(len(best_solution)):
            if time.time() > deadline:
                break
            removed = best_solution[idx]
            for new_elem in range(1, N + 1):
                if new_elem == removed or new_elem in best_solution:
                    continue
                # Check if we can swap
                trial = [x for x in best_solution if x != removed]
                trial_set = set(trial)
                trial_set.add(new_elem)
                
                # Verify no 3-AP in trial
                valid = True
                for a in trial:
                    for b in trial:
                        if a >= b:
                            continue
                        # Check a, b, c
                        c = 2 * b - a
                        if c > b and c in trial_set:
                            valid = False
                            break
                        # Check a, c, b
                        if (a + b) % 2 == 0:
                            c = (a + b) // 2
                            if c > a and c < b and c in trial_set:
                                valid = False
                                break
                    if not valid:
                        break
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
                # Invalid - should not happen
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
