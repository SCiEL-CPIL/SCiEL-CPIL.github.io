"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
        
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Convert to 0-based signed literals for internal use? 
    # The input is 1-based. 
    # Let's keep 1-based for simplicity in final output, 
    # but work with 0-based indices for speed? 
    # No, let's stick to the input format but optimize access.
    
    # --- Preprocessing: Unit Propagation and Pure Literal Elimination ---
    
    def simplify(clauses, assignment):
        """
        assignment: list of -1, 0, 1 (unassigned, false, true)
        Returns: (simplified_clauses, updated_assignment, conflict_flag)
        """
        # assignment is a list of ints: -1 (unassigned), 0 (False), 1 (True)
        # Clauses are lists of signed literals (1-based)
        
        changed = True
        while changed:
            changed = False
            new_clauses = []
            
            # 1. Process Unit Clauses and Tautologies
            # We need to check each clause
            # To make it efficient, we can scan clauses
            
            # First pass: Identify units and tautologies
            unit_vars = {} # var_idx -> value
            taut_clauses = set()
            
            for c in clauses:
                # Check for tautology or unit
                has_true = False
                has_false = False
                unassigned = []
                
                for lit in c:
                    var = abs(lit) - 1
                    val = assignment[var]
                    if val == 1:
                        if lit > 0:
                            has_true = True
                        else:
                            has_false = True
                    elif val == 0:
                        if lit > 0:
                            has_false = True
                        else:
                            has_true = True
                    else: # unassigned
                        unassigned.append(lit)
                
                if has_true and has_false:
                    # Tautology, remove
                    continue
                elif has_true:
                    # Satisfied, remove
                    continue
                elif has_false:
                    # Remove false literals
                    new_c = [lit for lit in unassigned]
                    if not new_c:
                        return None, assignment, True # Conflict
                    else:
                        new_clauses.append(new_c)
                else:
                    # No assigned literals
                    if len(unassigned) == 1:
                        # Unit clause
                        lit = unassigned[0]
                        var = abs(lit) - 1
                        val = 1 if lit > 0 else 0
                        if var in unit_vars and unit_vars[var] != val:
                            return None, assignment, True # Conflict
                        unit_vars[var] = val
                    else:
                        new_clauses.append(unassigned)
            
            # Apply unit assignments
            if unit_vars:
                for var, val in unit_vars.items():
                    if assignment[var] == -1:
                        assignment[var] = val
                        changed = True
                    elif assignment[var] != val:
                        return None, assignment, True # Conflict
                
                clauses = new_clauses
                continue # Restart loop to check for new units/pure literals
            
            # 2. Pure Literal Elimination
            # Find variables that appear in only one polarity in the remaining unassigned clauses
            pos_count = [0] * n
            neg_count = [0] * n
            
            for c in clauses:
                for lit in c:
                    var = abs(lit) - 1
                    if lit > 0:
                        pos_count[var] += 1
                    else:
                        neg_count[var] += 1
            
            pure_vars = {}
            for var in range(n):
                if assignment[var] == -1:
                    if pos_count[var] > 0 and neg_count[var] == 0:
                        pure_vars[var] = 1
                    elif neg_count[var] > 0 and pos_count[var] == 0:
                        pure_vars[var] = 0
            
            if pure_vars:
                for var, val in pure_vars.items():
                    assignment[var] = val
                    changed = True
                # We don't remove clauses immediately here, 
                # but the next iteration of the while loop will filter them out
                # Actually, we should filter them out to keep the list small
                # The next iteration of the while loop (changed=True) will re-process
                # But we just assigned them, so the next pass will remove satisfied clauses.
                # We can just continue the while loop.
                continue
            
            # No changes
            break
            
        return clauses, assignment, False

    # Initialize assignment: -1 for unassigned
    assignment = [-1] * n
    
    # Run simplification
    simplified_clauses, assignment, conflict = simplify(clauses, assignment)
    
    if conflict:
        # Conflict during simplification? Unlikely for satisfiable, but handle it.
        # Fall back to WalkSAT
        pass
    else:
        if simplified_clauses is None or len(simplified_clauses) == 0:
            # Check if all assigned?
            # If no clauses left, we are satisfied (assuming no conflicts)
            # But we need to return a full assignment.
            # Fill unassigned with False (or random)
            for i in range(n):
                if assignment[i] == -1:
                    assignment[i] = 0
            return [bool(x) for x in assignment]
        
        # If we still have clauses, proceed to WalkSAT
        clauses = simplified_clauses
        
    # --- WalkSAT ---
    
    # Re-index variables? 
    # The simplified clauses might only involve a subset of variables.
    # But we can just use the original n and the current assignment.
    
    # Pre-compute clause structure for fast updates
    # We need to map variable index to list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    clause_vars = []
    
    for ci, c in enumerate(clauses):
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))
            
    m = len(clauses)
    
    if m == 0:
        for i in range(n):
            if assignment[i] == -1:
                assignment[i] = 0
        return [bool(x) for x in assignment]

    def count_unsat(assignment):
        cnt = 0
        for ci, cv in enumerate(clause_vars):
            sat = False
            for vi, pos in cv:
                if assignment[vi] == (1 if pos else 0):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    def weighted_walksat(assignment, max_steps, deadline):
        rng = random.Random(int(time.time() * 1000) % 2**31)
        
        # Initialize unsat count
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return True
            
        unsat_clauses = []
        for ci, cv in enumerate(clause_vars):
            sat = False
            for vi, pos in cv:
                if assignment[vi] == (1 if pos else 0):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(ci)
        
        best_unsat = unsat_count
        stagnation = 0
        p = 0.55
        min_p = 0.3
        max_p = 0.9
        stagnation_threshold = 1000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return True
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                stagnation = 0
            else:
                stagnation += 1
                
            if stagnation > stagnation_threshold:
                p = min(max_p, p + 0.05)
            elif stagnation < 100:
                p = max(min_p, p - 0.01)
                
            if not unsat_clauses:
                return True
                
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            if rng.random() < p:
                vi, pos = rng.choice(cv)
            else:
                # Greedy
                best_delta = float('inf')
                best_vi = cv[0][0]
                for vi, pos in cv:
                    # Estimate delta
                    # How many clauses are satisfied by this var currently?
                    # And how many would be unsatisfied if we flip?
                    # Simple heuristic: count unsat clauses containing vi
                    delta_est = 0
                    for ci2, pos2 in var_clauses[vi]:
                        cv2 = clause_vars[ci2]
                        sat2 = False
                        for vi3, p3 in cv2:
                            if vi3 == vi:
                                continue
                            if assignment[vi3] == (1 if p3 else 0):
                                sat2 = True
                                break
                        if not sat2:
                            delta_est += 1
                            
                    if delta_est < best_delta:
                        best_delta = delta_est
                        best_vi = vi
                vi = best_vi
                
            # Flip
            old_val = assignment[vi]
            new_val = 1 - old_val
            assignment[vi] = new_val
            
            # Update unsat clauses
            # Only clauses containing vi are affected
            # We need to update unsat_clauses list
            # This is O(m) if we rebuild, but we can update incrementally
            # For simplicity and speed in Python, rebuilding is O(m * 3) which is fine for small m
            # But let's try to be smarter.
            
            new_unsat_clauses = []
            for ci2, cv2 in enumerate(clause_vars):
                # Check if clause contains vi
                contains_vi = False
                for vi3, _ in cv2:
                    if vi3 == vi:
                        contains_vi = True
                        break
                if contains_vi:
                    sat = False
                    for vi3, p3 in cv2:
                        if assignment[vi3] == (1 if p3 else 0):
                            sat = True
                            break
                    if not sat:
                        new_unsat_clauses.append(ci2)
                else:
                    # If it was unsat before, it remains unsat
                    # We don't have a "was_unsat" flag easily here without a set
                    # Let's just rebuild from scratch for correctness, 
                    # or maintain a set of unsat indices.
                    pass
            
            # Rebuilding is safer
            unsat_clauses = []
            for ci2, cv2 in enumerate(clause_vars):
                sat = False
                for vi3, p3 in cv2:
                    if assignment[vi3] == (1 if p3 else 0):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(ci2)
            
            unsat_count = len(unsat_clauses)
            
        return unsat_count == 0

    # Multi-restart
    best_assignment = None
    best_score = m + 1
    
    # If some vars are unassigned, assign them randomly
    for i in range(n):
        if assignment[i] == -1:
            assignment[i] = random.randint(0, 1)
            
    while time.time() < deadline:
        # Run WalkSAT
        found = weighted_walksat(assignment, 50000, deadline)
        
        if found:
            # Ensure all vars are assigned
            for i in range(n):
                if assignment[i] == -1:
                    assignment[i] = 0
            return [bool(x) for x in assignment]
            
        current_unsat = count_unsat(assignment)
        
        if current_unsat < best_score:
            best_score = current_unsat
            best_assignment = assignment[:]
            
        # If close to solution, try to polish
        if best_score <= 5 and time.time() < deadline:
            polish_rng = random.Random()
            for _ in range(10000):
                if time.time() > deadline:
                    break
                if best_score == 0:
                    break
                    
                unsat_clauses = []
                for ci, cv in enumerate(clause_vars):
                    sat = False
                    for vi, pos in cv:
                        if best_assignment[vi] == (1 if pos else 0):
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(ci)
                    
                if not unsat_clauses:
                    best_score = 0
                    break
                    
                ci = polish_rng.choice(unsat_clauses)
                cv = clause_vars[ci]
                vi = polish_rng.choice(cv)[0]
                
                old_val = best_assignment[vi]
                best_assignment[vi] = 1 - old_val
                
                new_unsat = count_unsat(best_assignment)
                
                if new_unsat < best_score:
                    best_score = new_unsat
                else:
                    best_assignment[vi] = old_val
                    
            if best_score == 0:
                for i in range(n):
                    if best_assignment[i] == -1:
                        best_assignment[i] = 0
                return [bool(x) for x in best_assignment]
                
        # Restart with random assignment
        assignment = [random.randint(0, 1) for _ in range(n)]
        
    if best_assignment is not None:
        for i in range(n):
            if best_assignment[i] == -1:
                best_assignment[i] = 0
        return [bool(x) for x in best_assignment]
        
    return [False] * n

# EVOLVE-BLOCK-END
