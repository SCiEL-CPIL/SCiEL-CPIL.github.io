"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def solve(instance):
    start_time = time.time()
    time_limit = 280.0  # Safety margin (300s kill, leave buffer)
    
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Preprocess: build clause variables and variable-to-clause index
    clause_vars = []  # list of lists of (var_idx, is_pos)
    var_to_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
            var_to_clauses[var_idx].append(i)
        clause_vars.append(cv)
    
    # For each clause, store the polarity of each variable for quick lookup
    # clause_polarity[i][var_idx] = is_pos or None
    clause_polarity = [{} for _ in range(m)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            clause_polarity[i][var_idx] = is_pos
    
    def is_satisfied_fast(clause_idx, assignment):
        cv = clause_vars[clause_idx]
        for var_idx, is_pos in cv:
            if assignment[var_idx] == is_pos:
                return True
        return False
    
    def count_unsat(assignment):
        count = 0
        for i in range(m):
            cv = clause_vars[i]
            sat = False
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    best_assignment = None
    best_unsat_count = float('inf')
    
    # WalkSAT with adaptive parameters
    rng = random.Random()
    
    def walk_sat():
        nonlocal best_assignment, best_unsat_count
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Build initial unsat set
        unsat_set = set()
        for i in range(m):
            if not is_satisfied_fast(i, assignment):
                unsat_set.add(i)
        
        if not unsat_set:
            return assignment
        
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # Adaptive walk parameters
        max_steps = 50000
        noise = 0.45  # Probability of random flip
        min_gain = -5  # Minimum acceptable damage
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            if not unsat_set:
                return assignment
            
            # Pick a random unsat clause
            idx = rng.choice(list(unsat_set))
            clause = clause_vars[idx]
            
            # Try to find a flip with minimum damage
            best_var_idx = -1
            best_damage = float('inf')
            best_is_pos = False
            
            # Evaluate each literal in the clause
            for var_idx, is_pos in clause:
                # Compute damage: how many currently-satisfied clauses become unsat
                damage = 0
                for c_idx in var_to_clauses[var_idx]:
                    if c_idx == idx:
                        continue
                    # Check if c_idx is currently satisfied
                    cv = clause_vars[c_idx]
                    is_sat_now = False
                    for v, p in cv:
                        if assignment[v] == p:
                            is_sat_now = True
                            break
                    
                    if not is_sat_now:
                        continue  # Already unsat, no damage
                    
                    # Check if c_idx remains satisfied after flipping var_idx
                    # The literal for var_idx in c_idx has polarity p_c
                    p_c = clause_polarity[c_idx].get(var_idx)
                    if p_c is None:
                        continue  # var_idx not in this clause (shouldn't happen)
                    
                    # After flip, assignment[var_idx] changes
                    # If the flipped literal was the only satisfied one, damage
                    # Check if any other literal in c_idx is satisfied
                    remains_sat = False
                    for v, p in cv:
                        if v == var_idx:
                            continue
                        if assignment[v] == p:
                            remains_sat = True
                            break
                    
                    if not remains_sat:
                        damage += 1
                
                if damage < best_damage:
                    best_damage = damage
                    best_var_idx = var_idx
                    best_is_pos = is_pos
            
            # Decide which variable to flip
            if best_var_idx == -1:
                var_idx, is_pos = rng.choice(clause)
            elif rng.random() < noise or best_damage > 10:
                # Random flip for exploration
                var_idx, is_pos = rng.choice(clause)
            else:
                var_idx = best_var_idx
                is_pos = best_is_pos
            
            # Perform flip
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Update unsat_set incrementally
            for c_idx in var_to_clauses[var_idx]:
                was_unsat = c_idx in unsat_set
                is_now_sat = is_satisfied_fast(c_idx, assignment)
                
                if was_unsat and is_now_sat:
                    unsat_set.remove(c_idx)
                elif not was_unsat and not is_now_sat:
                    unsat_set.add(c_idx)
            
            # Check for solution
            if not unsat_set:
                return assignment
            
            # Track best
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            # Adaptive noise: increase noise if stuck
            if step % 500 == 0 and step > 0:
                if current_unsat >= best_unsat_count:
                    noise = min(0.6, noise + 0.02)
                else:
                    noise = max(0.3, noise - 0.01)
        
        return None
    
    # Main loop with restarts
    restarts = 0
    max_restarts = 200
    
    while time.time() - start_time < time_limit and restarts < max_restarts:
        result = walk_sat()
        if result is not None:
            return result
        restarts += 1
        
        # If we have a good best, try to improve from it
        if best_assignment and best_unsat_count < m * 0.3:
            # Start from best assignment for next walk
            pass  # walk_sat starts fresh, but best is tracked
    
    if best_assignment:
        return best_assignment
    
    # Final fallback: return last assignment or random
    return [random.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
