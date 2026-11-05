import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 295
    
    # Pre-compute mappings
    # var_to_clauses[i] = list of clause indices
    var_to_clauses = [[] for _ in range(n)]
    # clause_vars[c] = list of var indices in clause c
    # clause_signs[c] = list of signs (+1/-1) in clause c
    clause_vars = []
    clause_signs = []
    
    for i, clause in enumerate(clauses):
        vars_in_clause = []
        signs_in_clause = []
        for l in clause:
            v = abs(l) - 1
            vars_in_clause.append(v)
            signs_in_clause.append(1 if l > 0 else -1)
            var_to_clauses[v].append(i)
        clause_vars.append(vars_in_clause)
        clause_signs.append(signs_in_clause)
        
    num_clauses = len(clauses)
    rng = random.Random()

    best_assignment = None
    best_unsat = float('inf')
    
    # VSIDS Activity
    activity = [1.0] * n
    decay_factor = 0.95
    
    # Helper to initialize sat_counts and unsat list from an assignment
    def init_state(assignment):
        sat_counts = [0] * num_clauses
        unsat_list = []
        is_unsat = [False] * num_clauses
        
        for c_idx in range(num_clauses):
            count = 0
            vars_c = clause_vars[c_idx]
            signs_c = clause_signs[c_idx]
            for j in range(3):
                v = vars_c[j]
                s = signs_c[j]
                if s == 1:
                    if assignment[v]:
                        count += 1
                else:
                    if not assignment[v]:
                        count += 1
            sat_counts[c_idx] = count
            if count == 0:
                unsat_list.append(c_idx)
                is_unsat[c_idx] = True
                
        return sat_counts, unsat_list, is_unsat

    # WalkSAT Loop
    restart = 0
    while time.time() < deadline - 5:
        # Luby-like restart length? 
        # For n=50, fixed 1000-2000 steps is often good. 
        # Let's use a slightly adaptive approach: increase steps if we are improving, keep same if stuck.
        steps_per_restart = 1000 + (restart % 5) * 200
        
        # Phase Saving: Use best_assignment if available, else random
        if best_assignment is not None:
            assignment = best_assignment[:]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        sat_counts, unsat_list, is_unsat = init_state(assignment)
        
        if not unsat_list:
            return assignment
            
        cur_unsat = len(unsat_list)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
        # Local Search
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            if not unsat_list:
                return assignment
                
            # Pick random unsat clause
            # O(1) selection from list
            c_idx = rng.randrange(len(unsat_list))
            clause_vars_c = clause_vars[c_idx]
            clause_signs_c = clause_signs[c_idx]
            
            # Evaluate flip for each literal in clause
            best_var = -1
            best_score = float('inf')
            
            for j in range(3):
                var_idx = clause_vars_c[j]
                sign_c = clause_signs_c[j]
                
                # Calculate breakage and fix
                # Flipping var_idx will fix c_idx (since it's unsat, this literal is false, flip makes it true)
                broken = 0
                fixed = 0
                
                for other_c_idx in var_to_clauses[var_idx]:
                    if other_c_idx == c_idx:
                        continue
                    
                    # Check if other_c_idx is satisfied
                    if not is_unsat[other_c_idx]:
                        # It is satisfied.
                        # It might break if it was satisfied ONLY by var_idx.
                        # sat_counts[other_c_idx] == 1 means exactly one literal is true.
                        if sat_counts[other_c_idx] == 1:
                            # Check if the literal for var_idx in other_c_idx is the true one.
                            # Find the index of var_idx in other_c_idx
                            # Since clauses are size 3, we can just loop through them.
                            o_vars = clause_vars[other_c_idx]
                            o_signs = clause_signs[other_c_idx]
                            # Check if var_idx is in this clause (it is, by definition of var_to_clauses)
                            # Determine current value of the literal
                            # We need to find which position var_idx is at.
                            # Optimization: Pre-compute var_to_pos_in_clause? 
                            # Or just loop 3 times.
                            is_true_literal = False
                            for k in range(3):
                                if o_vars[k] == var_idx:
                                    # Check if this literal is currently true
                                    if o_signs[k] == 1:
                                        if assignment[var_idx]:
                                            is_true_literal = True
                                    else:
                                        if not assignment[var_idx]:
                                            is_true_literal = True
                                    break
                            
                            if is_true_literal:
                                # This was the only true literal. Flipping makes it false.
                                # Clause becomes unsat.
                                broken += 1
                        # If sat_counts > 1, it won't break.
                    else:
                        # It is unsat (sat_counts == 0).
                        # All literals are false.
                        # Flipping var_idx makes the literal for var_idx true.
                        # So the clause becomes satisfied.
                        fixed += 1
                        
                # Score function
                # Minimize broken, maximize fixed
                # Add small activity bonus to break ties or prefer active vars
                score = broken - 0.5 * fixed + 0.01 * activity[var_idx]
                
                if score < best_score:
                    best_score = score
                    best_var = var_idx
                    
            # Decide flip
            # Noise parameter: Start with 0.5, maybe reduce if stuck?
            # Standard WalkSAT uses 0.5.
            p_noise = 0.5
            if rng.random() < p_noise:
                # Pick random literal from the unsat clause
                j = rng.randrange(3)
                var_idx = clause_vars_c[j]
            else:
                var_idx = best_var
                
            if var_idx == -1:
                continue
                
            # Perform flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update sat_counts and unsat status
            for c_idx in var_to_clauses[var_idx]:
                # Determine change in sat_count
                # We flipped var_idx.
                # Let's find the sign of var_idx in c_idx.
                o_vars = clause_vars[c_idx]
                o_signs = clause_signs[c_idx]
                
                # Find position of var_idx
                pos = -1
                for k in range(3):
                    if o_vars[k] == var_idx:
                        pos = k
                        break
                
                if pos == -1:
                    continue # Should not happen
                
                sign = o_signs[pos]
                # Old value of assignment[var_idx] was opposite of new.
                # Let's just recalculate the delta.
                # New value of literal:
                if sign == 1:
                    new_lit_val = 1 if assignment[var_idx] else 0
                else:
                    new_lit_val = 1 if not assignment[var_idx] else 0
                
                # Old value of literal was 1 - new_lit_val (since it flipped)
                old_lit_val = 1 - new_lit_val
                
                delta = new_lit_val - old_lit_val # +1 if became true, -1 if became false
                
                old_count = sat_counts[c_idx]
                new_count = old_count + delta
                sat_counts[c_idx] = new_count
                
                if old_count == 0 and new_count > 0:
                    # Became satisfied
                    is_unsat[c_idx] = False
                    # Remove from unsat_list
                    # Find index of c_idx in unsat_list
                    # Optimization: Keep a map clause_idx -> index in unsat_list?
                    # Or just linear search if list is small.
                    # Unsats can be up to M. M is usually ~ 4-5 * n = 200-250.
                    # Linear search O(M).
                    # Swap with last is better.
                    # We need to know the position.
                    # Let's maintain a dictionary unsat_pos = {clause_idx: index_in_list}
                    # Actually, updating a dictionary on every add/remove is overhead.
                    # Given M is small, linear search + pop might be fine.
                    # But let's try to be efficient.
                    # We'll use a list and a helper to remove.
                    # For now, simple linear scan to find and swap.
                    # This is O(M).
                    for i in range(len(unsat_list)):
                        if unsat_list[i] == c_idx:
                            unsat_list[i] = unsat_list[-1]
                            unsat_list.pop()
                            break
                            
                elif old_count > 0 and new_count == 0:
                    # Became unsat
                    is_unsat[c_idx] = True
                    unsat_list.append(c_idx)
                    
            # Update Activity
            activity[var_idx] += 1.0
            # Periodic decay
            if step % 100 == 0:
                for i in range(n):
                    activity[i] *= decay_factor

        # End local search
        restart += 1

    if best_assignment:
        return best_assignment
    return [False] * n
