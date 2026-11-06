import math
import time

def construct():
    deadline = time.time() + 110
    best_score = 0.0
    best_solution = []
    
    # Moduli from 20 to 200
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    weights = [1.0 / m for m in moduli]
    
    # Suffix sum of weights for upper bound
    # suffix_sum[i] is the sum of weights[i] to weights[n_mods-1]
    suffix_sum = [0.0] * (n_mods + 1)
    for i in range(n_mods - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
    
    # Global variables for B&B
    # selected is a list of (m, r)
    
    def get_valid_residues(m, selected):
        """
        Returns a list of valid residues r for modulus m,
        given a list of selected (m2, r2) pairs.
        """
        valid = []
        # Precompute constraints
        constraints = []
        for m2, r2 in selected:
            g = math.gcd(m, m2)
            constraints.append((g, r2 % g))
        
        for r in range(m):
            is_valid = True
            for g, r2_mod in constraints:
                if r % g == r2_mod:
                    is_valid = False
                    break
            if is_valid:
                valid.append(r)
        return valid

    def search(idx, current_sum, selected):
        nonlocal best_score, best_solution
        
        if time.time() > deadline:
            return

        # Upper bound: current sum + sum of all remaining weights
        upper_bound = current_sum + suffix_sum[idx]
        
        if upper_bound <= best_score:
            return

        if idx == n_mods:
            if current_sum > best_score:
                best_score = current_sum
                best_solution = selected[:]
            return

        m = moduli[idx]
        
        # Check if m is blocked
        valid_residues = get_valid_residues(m, selected)
        
        # Option 1: Skip m
        # We can prune the skip branch if even taking all remaining (including m) doesn't beat best
        # But we already checked upper_bound which includes m.
        # If we skip m, the new upper bound is current_sum + suffix_sum[idx+1]
        if current_sum + suffix_sum[idx + 1] > best_score:
            search(idx + 1, current_sum, selected)
        
        # Option 2: Include m
        if valid_residues:
            # To optimize, we can order residues. 
            # Residues that are "more central" or leave more options might be better.
            # For now, try all.
            for r in valid_residues:
                new_selected = selected + [(m, r)]
                search(idx + 1, current_sum + weights[idx], new_selected)
                if time.time() > deadline:
                    return

    # Initial call
    search(0, 0.0, [])
    
    # If no solution found (should not happen), return empty
    if not best_solution:
        # Fallback: greedy
        selected = []
        for m in moduli:
            valids = get_valid_residues(m, selected)
            if valids:
                selected.append((m, valids[0]))
        best_solution = selected
        
    return [[m, r] for m, r in best_solution]
