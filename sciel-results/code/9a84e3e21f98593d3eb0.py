"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from collections import deque

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 330s wall-clock kill, leave 10s margin
    deadline = time.time() + 320.0
    
    # Preprocess: convert clauses to 0-based indices and store signs
    # clause_vars[i] = array of variable indices (0-based)
    # clause_signs[i] = array of booleans (True if positive literal)
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_ = []
        signs = []
        for lit in c:
            v = abs(lit) - 1
            s = lit > 0
            vars_.append(v)
            signs.append(s)
        clause_vars.append(vars_)
        clause_signs.append(signs)
    
    # Variable occurrence counts for weighting
    var_pos_count = [0] * n
    var_neg_count = [0] * n
    for c in clauses:
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                var_pos_count[v] += 1
            else:
                var_neg_count[v] += 1
    
    # Build reverse mapping: variable -> list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for j in range(3):
            v = clause_vars[ci][j]
            s = clause_signs[ci][j]
            var_to_clauses[v].append((ci, s))
    
    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses."""
        for i in range(m):
            vars_ = clause_vars[i]
            signs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[vars_[j]] == signs[j]:
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for i in range(m):
            vars_ = clause_vars[i]
            signs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[vars_[j]] == signs[j]:
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def unsat_clauses(assignment):
        """Return list of indices of unsatisfied clauses."""
        unsat = []
        for i in range(m):
            vars_ = clause_vars[i]
            signs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[vars_[j]] == signs[j]:
                    sat = True
                    break
            if not sat:
                unsat.append(i)
        return unsat
    
    def dpll_solve(assignment, time_limit):
        """
        DPLL solver with unit propagation and pure literal elimination.
        Returns a satisfying assignment if found within time_limit, else None.
        """
        # Work on a copy of the assignment
        current = assignment[:]
        
        # Track which variables are unassigned (None) or assigned (True/False)
        # We'll use a dict for the DPLL search space
        # For efficiency, we track partial assignments as a list where None means unassigned
        
        # Build initial clause list
        # Each clause is a list of (var, sign) pairs
        # We'll track which literals are satisfied, falsified, or unassigned
        
        # For DPLL, we maintain:
        # - current assignment (partial)
        # - active clauses (those not yet satisfied)
        
        # Start with all clauses active
        active_clauses = list(range(m))
        
        # Unit propagation queue
        def propagate(assignment, active_clauses):
            """
            Perform unit propagation and pure literal elimination.
            Returns (success, new_assignment, new_active_clauses)
            success is False if a contradiction is found.
            """
            changed = True
            while changed:
                changed = False
                
                # Unit propagation: find clauses with exactly one unassigned literal
                new_active = []
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    
                    unassigned = []
                    satisfied = False
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if assignment[v] is None:
                            unassigned.append((v, s))
                        elif assignment[v] == s:
                            satisfied = True
                            break
                        # else: literal is falsified, continue
                    
                    if satisfied:
                        # Clause is satisfied, remove from active
                        continue
                    elif len(unassigned) == 0:
                        # All literals falsified: contradiction
                        return False, assignment, active_clauses
                    elif len(unassigned) == 1:
                        # Unit clause: assign the remaining literal
                        v, s = unassigned[0]
                        if assignment[v] is not None:
                            # Shouldn't happen, but just in case
                            continue
                        assignment[v] = s
                        changed = True
                        # Don't add to new_active since it's now satisfied
                    else:
                        new_active.append(ci)
                
                active_clauses = new_active
                
                # Pure literal elimination
                # For each unassigned variable, check if it appears only with one sign
                # in the active clauses
                pure_vars = {}
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if assignment[v] is None:
                            if v not in pure_vars:
                                pure_vars[v] = set()
                            pure_vars[v].add(s)
                
                for v, signs_set in pure_vars.items():
                    if len(signs_set) == 1:
                        # Pure literal: assign it
                        s = signs_set.pop()
                        assignment[v] = s
                        changed = True
                
                # Re-filter active clauses after pure literal assignment
                if changed:
                    new_active = []
                    for ci in active_clauses:
                        vars_ = clause_vars[ci]
                        signs = clause_signs[ci]
                        satisfied = False
                        for j in range(3):
                            v = vars_[j]
                            s = signs[j]
                            if assignment[v] is not None:
                                if assignment[v] == s:
                                    satisfied = True
                                    break
                                # else falsified
                            else:
                                # unassigned
                                pass
                        if not satisfied:
                            new_active.append(ci)
                    active_clauses = new_active
            
            return True, assignment, active_clauses
        
        # Start DPLL search
        # We'll use a simple recursive approach with backtracking
        # For 50 variables, this should be fast
        
        # Initial assignment: all None
        dpll_assign = [None] * n
        # Pre-fill with the given assignment where it's not None
        # Actually, for DPLL, we start fresh but can use the heuristic assignment as a guide
        
        # Let's just start with all unassigned
        dpll_assign = [None] * n
        
        # First, try propagation with no initial assignments
        success, dpll_assign, active_clauses = propagate(dpll_assign, list(range(m)))
        
        if not success:
            return None
        
        # Check if all clauses are satisfied
        if not active_clauses:
            # All clauses satisfied, fill in remaining unassigned variables
            for v in range(n):
                if dpll_assign[v] is None:
                    dpll_assign[v] = assignment[v] if assignment[v] is not None else False
            return dpll_assign
        
        # Need to do DPLL search
        # Pick a variable to branch on
        # Use the variable that appears most in active clauses
        var_count = {}
        for ci in active_clauses:
            vars_ = clause_vars[ci]
            signs = clause_signs[ci]
            for j in range(3):
                v = vars_[j]
                if dpll_assign[v] is None:
                    var_count[v] = var_count.get(v, 0) + 1
        
        if not var_count:
            # No unassigned variables in active clauses, should be satisfied
            for v in range(n):
                if dpll_assign[v] is None:
                    dpll_assign[v] = assignment[v] if assignment[v] is not None else False
            return dpll_assign
        
        # Pick the variable with highest count
        branch_var = max(var_count, key=var_count.get)
        
        # Try both values
        for val in [True, False]:
            if time.time() > time_limit:
                return None
            
            # Save state
            saved_assign = dpll_assign[:]
            saved_active = active_clauses[:]
            
            # Assign and propagate
            dpll_assign[branch_var] = val
            success, dpll_assign, new_active = propagate(dpll_assign, saved_active)
            
            if success:
                if not new_active:
                    # All clauses satisfied
                    for v in range(n):
                        if dpll_assign[v] is None:
                            dpll_assign[v] = assignment[v] if assignment[v] is not None else False
                    return dpll_assign
                else:
                    # Recurse
                    result = dpll_solve_with_state(dpll_assign, new_active, assignment, time_limit)
                    if result is not None:
                        return result
            
            # Restore state
            dpll_assign = saved_assign
            active_clauses = saved_active
        
        return None
    
    def dpll_solve_with_state(current_assign, active_clauses, heuristic_assign, time_limit):
        """Helper for recursive DPLL."""
        if time.time() > time_limit:
            return None
        
        # Propagate
        changed = True
        while changed:
            changed = False
            new_active = []
            for ci in active_clauses:
                vars_ = clause_vars[ci]
                signs = clause_signs[ci]
                
                unassigned = []
                satisfied = False
                for j in range(3):
                    v = vars_[j]
                    s = signs[j]
                    if current_assign[v] is None:
                        unassigned.append((v, s))
                    elif current_assign[v] == s:
                        satisfied = True
                        break
                
                if satisfied:
                    continue
                elif len(unassigned) == 0:
                    return None
                elif len(unassigned) == 1:
                    v, s = unassigned[0]
                    current_assign[v] = s
                    changed = True
                else:
                    new_active.append(ci)
            
            active_clauses = new_active
            
            # Pure literal elimination
            pure_vars = {}
            for ci in active_clauses:
                vars_ = clause_vars[ci]
                signs = clause_signs[ci]
                for j in range(3):
                    v = vars_[j]
                    s = signs[j]
                    if current_assign[v] is None:
                        if v not in pure_vars:
                            pure_vars[v] = set()
                        pure_vars[v].add(s)
            
            for v, signs_set in pure_vars.items():
                if len(signs_set) == 1:
                    s = signs_set.pop()
                    current_assign[v] = s
                    changed = True
            
            if changed:
                new_active = []
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    satisfied = False
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if current_assign[v] is not None:
                            if current_assign[v] == s:
                                satisfied = True
                                break
                    if not satisfied:
                        new_active.append(ci)
                active_clauses = new_active
        
        if not active_clauses:
            for v in range(n):
                if current_assign[v] is None:
                    current_assign[v] = heuristic_assign[v] if heuristic_assign[v] is not None else False
            return current_assign
        
        # Pick branch variable
        var_count = {}
        for ci in active_clauses:
            vars_ = clause_vars[ci]
            for j in range(3):
                v = vars_[j]
                if current_assign[v] is None:
                    var_count[v] = var_count.get(v, 0) + 1
        
        if not var_count:
            for v in range(n):
                if current_assign[v] is None:
                    current_assign[v] = heuristic_assign[v] if heuristic_assign[v] is not None else False
            return current_assign
        
        branch_var = max(var_count, key=var_count.get)
        
        for val in [True, False]:
            if time.time() > time_limit:
                return None
            
            saved_assign = current_assign[:]
            
            current_assign[branch_var] = val
            
            # Propagate
            changed = True
            while changed:
                changed = False
                new_active = []
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    
                    unassigned = []
                    satisfied = False
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if current_assign[v] is None:
                            unassigned.append((v, s))
                        elif current_assign[v] == s:
                            satisfied = True
                            break
                    
                    if satisfied:
                        continue
                    elif len(unassigned) == 0:
                        current_assign = saved_assign
                        break
                    elif len(unassigned) == 1:
                        v, s = unassigned[0]
                        current_assign[v] = s
                        changed = True
                    else:
                        new_active.append(ci)
                
                if current_assign is not saved_assign and len(new_active) != len(active_clauses):
                    active_clauses = new_active
                
                # Pure literal elimination
                pure_vars = {}
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    for j in range(3):
                        v = vars_[j]
                        s = signs[j]
                        if current_assign[v] is None:
                            if v not in pure_vars:
                                pure_vars[v] = set()
                            pure_vars[v].add(s)
                
                for v, signs_set in pure_vars.items():
                    if len(signs_set) == 1:
                        s = signs_set.pop()
                        current_assign[v] = s
                        changed = True
                
                if changed:
                    new_active = []
                    for ci in active_clauses:
                        vars_ = clause_vars[ci]
                        signs = clause_signs[ci]
                        satisfied = False
                        for j in range(3):
                            v = vars_[j]
                            s = signs[j]
                            if current_assign[v] is not None:
                                if current_assign[v] == s:
                                    satisfied = True
                                    break
                        if not satisfied:
                            new_active.append(ci)
                    active_clauses = new_active
            
            if current_assign is not saved_assign:
                if not active_clauses:
                    for v in range(n):
                        if current_assign[v] is None:
                            current_assign[v] = heuristic_assign[v] if heuristic_assign[v] is not None else False
                    return current_assign
                else:
                    result = dpll_solve_with_state(current_assign, active_clauses, heuristic_assign, time_limit)
                    if result is not None:
                        return result
            
            # Restore
            current_assign = saved_assign
        
        return None
    
    best_assignment = None
    best_unsat = float('inf')
    
    rng = random.Random()
    
    # Strategy:
    # 1. Try DPLL first with a time limit of 50s
    # 2. If DPLL fails, fall back to WalkSAT with restarts
    # 3. Use DPLL as a polish step for near-satisfying assignments
    
    # Phase 1: DPLL with time limit
    dpll_deadline = time.time() + 50.0
    dpll_result = dpll_solve([None] * n, dpll_deadline)
    if dpll_result is not None:
        if is_satisfied(dpll_result):
            return dpll_result
    
    # Phase 2: WalkSAT with restarts
    max_restarts = 500
    steps_per_restart = 10000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 1.0:
            break
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # WalkSAT with adaptive noise
        for step in range(steps_per_restart):
            if time.time() > deadline - 1.0:
                break
            
            # Find unsatisfied clauses
            unsat = unsat_clauses(assignment)
            if not unsat:
                # Found satisfying assignment
                return assignment
            
            # Track best
            if len(unsat) < best_unsat:
                best_unsat = len(unsat)
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat)
            vars_ = clause_vars[clause_idx]
            signs = clause_signs[clause_idx]
            
            # Count how many literals in this clause are false
            # and how many true
            false_vars = []
            for j in range(3):
                if assignment[vars_[j]] != signs[j]:
                    false_vars.append(vars_[j])
            
            # Pick a variable to flip from the false literals
            if false_vars:
                v = rng.choice(false_vars)
            else:
                # This shouldn't happen if clause is unsatisfied, but just in case
                v = rng.choice(vars_)
            
            # Flip the variable
            assignment[v] = not assignment[v]
        
        # If we didn't find a solution in this restart, continue
    
    # Phase 3: If we have a near-satisfying assignment, try DPLL polish
    if best_assignment is not None and best_unsat <= 5:
        polish_deadline = time.time() + 30.0
        polish_result = dpll_solve(best_assignment, polish_deadline)
        if polish_result is not None:
            if is_satisfied(polish_result):
                return polish_result
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: random assignment
        return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
