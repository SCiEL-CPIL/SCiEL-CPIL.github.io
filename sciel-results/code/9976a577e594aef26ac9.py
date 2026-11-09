import time
import random
import sys
from collections import deque

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin for safety
    
    # Preprocessing: Convert clauses to (var, sign)
    # var is 0-indexed, sign is 1 or -1
    clause_data = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        c_data = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            c_data.append((var, sign))
            var_to_clauses[var].append(i)
        clause_data.append(c_data)

    # Helper to check if a clause is satisfied given current assignment
    def is_sat(clause_idx, assignment):
        for var, sign in clause_data[clause_idx]:
            if assignment[var] == (sign == 1):
                return True
        return False

    # DPLL solver with unit propagation and VSIDS
    def dpll_solve(assignment, trail, decision_level):
        """
        DPLL with unit propagation.
        assignment: list of None (unassigned) or True/False
        trail: list of (var, value, level)
        decision_level: current decision level
        Returns: True if solved, False if conflict, None if need backtracking
        """
        # Unit propagation
        changed = True
        while changed:
            changed = False
            for i in range(m):
                # Check if clause is unit or empty
                num_false = 0
                num_unassigned = 0
                unassigned_vars = []
                
                for var, sign in clause_data[i]:
                    val = assignment[var]
                    if val is None:
                        num_unassigned += 1
                        unassigned_vars.append((var, sign))
                    elif val == (sign == 1):
                        # Clause is satisfied
                        break
                    else:
                        num_false += 1
                
                if num_unassigned == 0:
                    if num_false == len(clause_data[i]):
                        # Conflict: clause is falsified
                        return False
                    else:
                        # Clause is satisfied
                        continue
                elif num_unassigned == 1:
                    # Unit clause: must assign the remaining variable
                    var, sign = unassigned_vars[0]
                    required_val = (sign == 1)
                    if assignment[var] != required_val:
                        assignment[var] = required_val
                        trail.append((var, required_val, decision_level))
                        changed = True
        
        # Check if all variables are assigned
        if all(v is not None for v in assignment):
            return True
        
        # Select variable using VSIDS
        # Simple VSIDS: pick variable with highest activity among unassigned
        # For simplicity, use a heuristic: pick variable that appears in most unsatisfied clauses
        var_scores = [0] * n
        for i in range(m):
            # Check if clause is unsatisfied (all literals false or unassigned)
            is_unsat = True
            for var, sign in clause_data[i]:
                val = assignment[var]
                if val is None:
                    is_unsat = False
                    break
                elif val == (sign == 1):
                    is_unsat = False
                    break
            
            if is_unsat:
                for var, sign in clause_data[i]:
                    if assignment[var] is None:
                        var_scores[var] += 1
        
        # Pick variable with highest score
        max_score = -1
        best_var = -1
        for var in range(n):
            if assignment[var] is None and var_scores[var] > max_score:
                max_score = var_scores[var]
                best_var = var
        
        if best_var == -1:
            # No unassigned variables but not all satisfied? Shouldn't happen
            return True
        
        # Try both values, prefer the one that satisfies more clauses
        # Count how many clauses would be satisfied by each value
        score_true = 0
        score_false = 0
        for i in range(m):
            for var, sign in clause_data[i]:
                if var == best_var:
                    if sign == 1:
                        score_true += 1
                    else:
                        score_false += 1
                    break
        
        # Try the better value first
        if score_true >= score_false:
            try_order = [True, False]
        else:
            try_order = [False, True]
        
        for val in try_order:
            assignment[best_var] = val
            trail.append((best_var, val, decision_level + 1))
            
            result = dpll_solve(assignment, trail, decision_level + 1)
            if result is True:
                return True
            elif result is False:
                # Backtrack
                assignment[best_var] = None
                # Pop trail entries for this decision level
                while trail and trail[-1][2] == decision_level + 1:
                    v, _, _ = trail.pop()
                    assignment[v] = None
        
        return None

    # Try DPLL first with a time budget
    dpll_budget = 60.0  # 60 seconds for DPLL
    dpll_start = time.time()
    
    # Initialize assignment with None
    dpll_assignment = [None] * n
    dpll_result = dpll_solve(dpll_assignment, [], 0)
    
    if dpll_result is True:
        # Convert to boolean list
        return [bool(v) for v in dpll_assignment]
    
    dpll_elapsed = time.time() - dpll_start
    
    # If DPLL didn't solve, fall back to WalkSAT
    best_assignment = None
    best_unsat = m + 1
    
    # Restart loop
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Random initialization
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Calculate initial unsat count
        unsat_count = 0
        clause_sat = [False] * m
        for i in range(m):
            if is_sat(i, assignment):
                clause_sat[i] = True
            else:
                unsat_count += 1
        
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        # WalkSAT search
        step_limit = 100 * n
        p_random = 0.3  # 30% chance to pick a random variable from the clause
        
        for step in range(step_limit):
            if time.time() > deadline - 5.0:
                return best_assignment
            
            if unsat_count == 0:
                return assignment

            # Pick a random unsatisfied clause
            unsat_clauses = [i for i in range(m) if not clause_sat[i]]
            if not unsat_clauses:
                return assignment
                
            c_idx = random.choice(unsat_clauses)
            
            # Calculate gain for each variable in this clause
            best_gain = -1000
            best_var = -1
            best_sign = 0
            
            # With probability p_random, pick a random variable from the clause
            if random.random() < p_random:
                var, sign = random.choice(clause_data[c_idx])
                best_var = var
                best_sign = sign
            else:
                for var, sign in clause_data[c_idx]:
                    # Calculate gain of setting assignment[var] = (sign == 1)
                    new_val = (sign == 1)
                    
                    gain = 0
                    # Check all clauses containing var
                    for c2_idx in var_to_clauses[var]:
                        is_c2_sat = clause_sat[c2_idx]
                        
                        # Find sign of var in c2
                        sign_in_c2 = None
                        for v2, s2 in clause_data[c2_idx]:
                            if v2 == var:
                                sign_in_c2 = s2
                                break
                                
                        if sign_in_c2 is None:
                            continue
                            
                        current_lit_true = (assignment[var] == (sign_in_c2 == 1))
                        new_lit_true = (new_val == (sign_in_c2 == 1))
                        
                        if is_c2_sat:
                            if current_lit_true and not new_lit_true:
                                other_true = False
                                for v2, s2 in clause_data[c2_idx]:
                                    if v2 != var:
                                        if assignment[v2] == (s2 == 1):
                                            other_true = True
                                            break
                                if not other_true:
                                    gain -= 1
                        else:
                            if not current_lit_true and new_lit_true:
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                        best_sign = sign
            
            # Flip best_var
            if best_var != -1:
                assignment[best_var] = (best_sign == 1)
                
                # Update clause_sat and unsat_count
                for c2_idx in var_to_clauses[best_var]:
                    was_sat = clause_sat[c2_idx]
                    is_now_sat = is_sat(c2_idx, assignment)
                    
                    if was_sat and not is_now_sat:
                        unsat_count += 1
                    elif not was_sat and is_now_sat:
                        unsat_count -= 1
                    
                    clause_sat[c2_idx] = is_now_sat

                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]

    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback
        return [False] * n
