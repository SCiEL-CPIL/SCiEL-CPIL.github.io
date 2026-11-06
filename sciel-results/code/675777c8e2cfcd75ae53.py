"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np

def solve(instance):
    start_time = time.time()
    deadline = start_time + 290.0
    
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Preprocess: Convert to 0-based, store signs as 1/-1
    # clause_vars[i] = list of (var_idx, sign) where sign is 1 or -1
    # assignment is an array of 1/-1
    # Clause satisfied if sum(sign * assignment[var]) > 0 for any literal? No, OR means if ANY literal is True.
    # Literal (var, sign) is True if assignment[var] == sign.
    
    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            sign = 1 if is_pos else -1
            cv.append((var_idx, sign))
            var_to_clauses[var_idx].append((i, sign))
        clause_vars.append(cv)
        
    # Convert to arrays for potential vectorization or just faster access?
    # List of tuples is okay.
    
    def is_satisfied(ci, assignment):
        # assignment is list or array of 1/-1
        for var_idx, sign in clause_vars[ci]:
            if assignment[var_idx] == sign:
                return True
        return False
        
    def count_unsat(assignment):
        count = 0
        for i in range(m):
            if not is_satisfied(i, assignment):
                count += 1
        return count

    # Precompute clause literals as arrays for faster iteration?
    # clause_vars is list of lists of tuples.
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    rng = random.Random(0)
    
    # Adaptive WalkSAT
    # State: assignment (list of 1/-1), unsat_clauses (list of indices), is_unsat (bool array)
    
    def fast_walksat(assignment, max_steps, noise, base_temp):
        # assignment is modified in place
        # Initialize unsat tracking
        unsat_list = []
        is_unsat = [False] * m
        for i in range(m):
            if not is_satisfied(i, assignment):
                unsat_list.append(i)
                is_unsat[i] = True
                
        if not unsat_list:
            return assignment, True, 0
            
        current_unsat = len(unsat_list)
        
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False, current_unsat
                
            if not unsat_list:
                return assignment, True, 0
                
            # Pick random unsat clause
            ci = rng.choice(unsat_list)
            clause = clause_vars[ci]
            
            # Choose variable to flip
            # Heuristic: Minimize damage (clauses that become unsat)
            # Damage = number of currently satisfied clauses that contain var with opposite sign
            # and have no other satisfied literals.
            # Approximation: Count satisfied clauses with opposite sign.
            
            if rng.random() < noise:
                var_idx, sign = rng.choice(clause)
                new_val = sign
            else:
                best_var = -1
                best_damage = 10**9
                best_val = 1
                
                for var_idx, sign in clause:
                    new_val = sign
                    old_val = assignment[var_idx]
                    if new_val == old_val:
                        continue # No change needed? But it's unsat, so this literal is false.
                        # Wait, if clause is unsat, ALL literals are false.
                        # So assignment[var_idx] != sign.
                        # So new_val (sign) != old_val.
                    
                    # Calculate damage
                    # Clauses containing var_idx with sign == -new_val (opposite)
                    # that are currently satisfied.
                    damage = 0
                    opposite_sign = -new_val
                    
                    for (oc, opos) in var_to_clauses[var_idx]:
                        if oc == ci:
                            continue
                        if opos == opposite_sign:
                            # This clause has the literal that will become false.
                            # If it is currently satisfied, it might become unsat.
                            # Check if it is satisfied ONLY by this literal.
                            # If is_unsat[oc] is False, it is satisfied.
                            if not is_unsat[oc]:
                                # Check if other literals in oc are satisfied
                                # If yes, no damage. If no, damage.
                                # Optimization: If we just count, it's an approximation.
                                # Let's do exact check for better quality, but it's slower.
                                # For 3-SAT, clause size is 3.
                                # Check other 2 literals.
                                remains_sat = False
                                for v, s in clause_vars[oc]:
                                    if v == var_idx:
                                        continue
                                    if assignment[v] == s:
                                        remains_sat = True
                                        break
                                if not remains_sat:
                                    damage += 1
                                    if damage >= best_damage:
                                        break
                    
                    if damage < best_damage:
                        best_damage = damage
                        best_var = var_idx
                        best_val = new_val
                        
                if best_var == -1:
                    # Fallback
                    var_idx, sign = rng.choice(clause)
                    new_val = sign
                else:
                    var_idx = best_var
                    new_val = best_val
            
            old_val = assignment[var_idx]
            if old_val == new_val:
                continue
                
            assignment[var_idx] = new_val
            
            # Update unsat status for clauses containing var_idx
            for (oci, opos) in var_to_clauses[var_idx]:
                if is_unsat[oci]:
                    # Was unsat, check if now sat
                    if is_satisfied(oci, assignment):
                        is_unsat[oci] = False
                        # Remove from unsat_list
                        # Efficient removal: swap with last and pop
                        # Find index
                        try:
                            idx = unsat_list.index(oci)
                            unsat_list[idx] = unsat_list[-1]
                            unsat_list.pop()
                            current_unsat -= 1
                        except ValueError:
                            pass
                else:
                    # Was sat, check if now unsat
                    if not is_satisfied(oci, assignment):
                        is_unsat[oci] = True
                        unsat_list.append(oci)
                        current_unsat += 1
                        
        return assignment, len(unsat_list) == 0, current_unsat

    # Strategy:
    # 1. Try a few random restarts with WalkSAT.
    # 2. If not solved, try a "polish" phase with lower noise and higher steps on the best found assignment.
    
    base_steps = 2000
    base_noise = 0.25
    max_restarts = 500
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
        
        # Adaptive noise: start high, decrease
        noise = base_noise
        steps = base_steps
        
        assignment, solved, final_unsat = fast_walksat(assignment, steps, noise, 0)
        
        if solved:
            return assignment
            
        if final_unsat < best_unsat_count:
            best_unsat_count = final_unsat
            best_assignment = assignment[:]
            
        # If we found a good assignment, try to polish it
        if r > 0 and best_assignment is not None and r % 10 == 0:
            if time.time() > deadline - 5:
                break
            # Polish: lower noise, more steps
            polish_assignment = best_assignment[:]
            polish_assignment, solved, final_unsat = fast_walksat(polish_assignment, 5000, 0.05, 0)
            if solved:
                return polish_assignment
            if final_unsat < best_unsat_count:
                best_unsat_count = final_unsat
                best_assignment = polish_assignment[:]

    # Final attempt on best assignment with very low noise
    if best_assignment is not None:
        if time.time() < deadline - 2:
            final_assignment = best_assignment[:]
            final_assignment, solved, _ = fast_walksat(final_assignment, 10000, 0.01, 0)
            if solved:
                return final_assignment
            if count_unsat(final_assignment) < best_unsat_count:
                best_assignment = final_assignment

    # Return best assignment found (might not be satisfying)
    # The task says "Return a satisfying assignment... score is fraction solved".
    # If we return an unsatisfying assignment, it scores 0 for that instance.
    # But we must return something.
    
    # Convert to booleans for return
    if best_assignment is None:
        return [False] * n
        
    return [val == 1 for val in best_assignment]

# EVOLVE-BLOCK-END
