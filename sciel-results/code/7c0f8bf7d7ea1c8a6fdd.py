"""Simulated Annealing based 3-SAT solver with conflict-aware variable selection."""
# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Pre-compute variable to clause indices
    var_to_clauses = [[] for _ in range(n)]
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_to_clauses[var_idx].append((c_idx, sign))

    # Random initial assignment
    assignment = [rng.randint(0, 1) for _ in range(n)]
    
    # Track satisfaction count for each clause
    num_clauses = len(clauses)
    sat_count = [0] * num_clauses
    unsat_clauses = []
    
    # Initialize sat_count and unsat_clauses
    for c_idx, clause in enumerate(clauses):
        count = 0
        for lit in clause:
            if assignment[abs(lit) - 1] == (1 if lit > 0 else 0):
                count += 1
        sat_count[c_idx] = count
        if count == 0:
            unsat_clauses.append(c_idx)
            
    current_energy = len(unsat_clauses)
    
    if current_energy == 0:
        return [bool(x) for x in assignment]

    # Simulated Annealing parameters
    T_start = 5.0
    T_end = 0.01
    max_iterations = 5000000
    
    cooling_rate = (T_end / T_start) ** (1.0 / max_iterations)
    T = T_start
    
    for iteration in range(max_iterations):
        if current_energy == 0:
            break
        
        # Pick a random unsatisfied clause
        # Filter unsat_clauses to be safe if we haven't cleaned recently
        # But we will clean every iteration for simplicity and correctness, 
        # or just pick from the list and verify.
        # Given the small size, picking from the list and verifying is fine.
        # However, if the list contains stale entries, we might pick a satisfied clause.
        # Let's maintain the list strictly.
        
        # To ensure unsat_clauses is accurate, we update it whenever sat_count changes.
        # But removing from a list is O(N). 
        # Let's use a list and just pick a random index, then check if it's still unsat.
        # If it is, use it. If not, pick again (or clean the list).
        # Since N is small, let's just rebuild the list every K steps or every step?
        # Rebuilding every step is O(C) ~ 500 ops. 5M steps * 500 ops = 2.5B ops. Too slow in Python.
        
        # Better: Use a set for O(1) add/remove, and convert to list for random choice?
        # Or just keep a list and clean it periodically.
        
        # Let's try a hybrid: Keep a list. When picking, if the clause is satisfied, 
        # remove it from the list (O(N) removal) and pick again.
        # Or simpler: Just iterate through unsat_clauses and pick a random valid one.
        
        # Let's optimize the inner loop.
        # We will maintain `unsat_clauses` as a list.
        # When a clause becomes satisfied, we don't remove it immediately.
        # When a clause becomes unsatisfied, we append it.
        # This means `unsat_clauses` might have duplicates and stale entries.
        # We will filter it every 100 iterations.
        
        if iteration % 100 == 0:
            unsat_clauses = [c for c in unsat_clauses if sat_count[c] == 0]
            # Remove duplicates
            unsat_clauses = list(set(unsat_clauses))
            current_energy = len(unsat_clauses)
            if current_energy == 0:
                break

        if not unsat_clauses:
             # Should not happen if current_energy > 0 and we just cleaned
             # But if we didn't clean, it might be empty of valid ones but list not empty?
             # No, if current_energy > 0, there is at least one unsat clause.
             # If unsat_clauses is empty, it means we missed adding one?
             # Let's just break if list is empty.
             break

        c_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
        
        # Verify it is actually unsatisfied (in case of stale entry before cleanup)
        if sat_count[c_idx] != 0:
            continue

        clause = clauses[c_idx]
        lit = clause[rng.randint(0, 2)]
        var = abs(lit) - 1
        
        old_val = assignment[var]
        new_val = 1 - old_val
        
        # Calculate delta energy
        delta = 0
        # We need to update sat_count for all clauses containing var
        # And determine delta based on changes
        
        affected_clauses = var_to_clauses[var]
        
        for c_idx_affected, sign in affected_clauses:
            old_sat = sat_count[c_idx_affected]
            
            is_satisfied_old = (old_val == 1) if sign == 1 else (old_val == 0)
            is_satisfied_new = (new_val == 1) if sign == 1 else (new_val == 0)
            
            if is_satisfied_old and not is_satisfied_new:
                # Clause loses a satisfied literal
                sat_count[c_idx_affected] -= 1
                if old_sat == 1 and sat_count[c_idx_affected] == 0:
                    # Clause becomes unsatisfied
                    delta += 1
            elif not is_satisfied_old and is_satisfied_new:
                # Clause gains a satisfied literal
                sat_count[c_idx_affected] += 1
                if old_sat == 0 and sat_count[c_idx_affected] == 1:
                    # Clause becomes satisfied
                    delta -= 1
            else:
                # Status of this literal didn't change satisfaction, or both changed (impossible for single var flip in 3-SAT? No, just one literal changes)
                # Wait, if is_satisfied_old == is_satisfied_new, sat_count doesn't change.
                pass

        # Accept or reject
        accepted = False
        if delta <= 0:
            accepted = True
        else:
            if T > 0:
                if rng.random() < math.exp(-delta / T):
                    accepted = True
        
        if accepted:
            assignment[var] = new_val
            current_energy += delta
        else:
            # Rollback sat_count changes
            for c_idx_affected, sign in affected_clauses:
                is_satisfied_old = (old_val == 1) if sign == 1 else (old_val == 0)
                is_satisfied_new = (new_val == 1) if sign == 1 else (new_val == 0)
                
                if is_satisfied_old and not is_satisfied_new:
                    sat_count[c_idx_affected] += 1 # Rollback decrement
                elif not is_satisfied_old and is_satisfied_new:
                    sat_count[c_idx_affected] -= 1 # Rollback increment

        # Cool down
        T *= cooling_rate
        
    return [bool(x) for x in assignment]
# EVOLVE-BLOCK-END
