"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 300s wall-clock limit, use 295s for safety margin.
    deadline = time.time() + 295.0

    # Convert clauses to 0-based signed literals
    # Input: 1-based signed. 1..n is pos, -1..-n is neg.
    # Output: 0-based signed. 0..n-1 is pos, -(0..n-1) is neg? 
    # Standard: lit > 0 means var lit-1 is True. lit < 0 means var -lit-1 is False.
    # Let's use: lit > 0 -> var = lit - 1, pol = 1. lit < 0 -> var = -lit - 1, pol = -1.
    
    clauses_v = []
    for c in clauses:
        cv = []
        for l in c:
            if l > 0:
                cv.append((l - 1, 1))
            else:
                cv.append((-l - 1, -1))
        clauses_v.append(cv)
        
    m = len(clauses_v)
    
    # Build index: for each variable, list of (clause_idx, polarity)
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_v):
        for var, pol in c:
            var_to_clauses[var].append((ci, pol))

    # Phase Saving: Initially all False, updated during search
    # We will pass this state around or use a global mutable one with backtracking
    # For DPLL, we will use a recursive function that passes `assignment` and `phase`
    
    def dpll_solve(deadline):
        """DPLL solver with unit propagation, pure literal elimination, and phase saving."""
        
        # assignment[var] = True/False/None (unassigned)
        assignment = [None] * n
        # phase[var] = suggested value for branching (True/False)
        phase = [False] * n
        
        # Track active clauses? 
        # Instead of filtering clauses, we can just check if a clause is satisfied/falsified on the fly.
        # For n=50, checking all clauses is fast.
        
        def is_clause_sat(ci, assignment):
            c = clauses_v[ci]
            for var, pol in c:
                if assignment[var] is None:
                    continue
                if assignment[var] == (pol == 1):
                    return True
            return False

        def is_clause_falsified(ci, assignment):
            c = clauses_v[ci]
            for var, pol in c:
                if assignment[var] is None:
                    return False
                if assignment[var] != (pol == 1):
                    return False
                # else literal is true, so clause is not falsified
            return True

        def get_unsat_clauses(assignment):
            """Return list of indices of clauses that are not satisfied."""
            unsat = []
            for ci in range(m):
                if not is_clause_sat(ci, assignment):
                    # Check if falsified
                    if is_clause_falsified(ci, assignment):
                        return None # Contradiction
                    unsat.append(ci)
            return unsat

        def propagate_units(assignment, phase):
            """
            Iteratively assign variables forced by unit clauses.
            Returns False if contradiction, True otherwise.
            Modifies assignment and phase in place.
            """
            changed = True
            while changed:
                changed = False
                for ci in range(m):
                    c = clauses_v[ci]
                    # Check if clause is already satisfied
                    sat = False
                    unassigned_count = 0
                    unit_var = None
                    unit_pol = None
                    falsified = True
                    
                    for var, pol in c:
                        val = assignment[var]
                        if val is None:
                            unassigned_count += 1
                            falsified = False
                            if unassigned_count == 1:
                                unit_var = var
                                unit_pol = pol
                        else:
                            if val == (pol == 1):
                                sat = True
                                break
                            # else literal is false
            
                    if sat:
                        continue
                    
                    if unassigned_count == 0:
                        # Clause is falsified
                        return False
                    
                    if unassigned_count == 1:
                        # Unit clause
                        if assignment[unit_var] is None:
                            assignment[unit_var] = (unit_pol == 1)
                            phase[unit_var] = (unit_pol == 1)
                            changed = True
            
            return True

        def pure_literal_elimination(assignment, phase):
            """
            Assign pure literals.
            Returns False if contradiction (should not happen for pure), True otherwise.
            """
            # Find variables that appear only with one polarity in unsatisfied clauses
            # This is O(n * m) which is fine for n=50.
            
            # First, identify unsatisfied clauses
            unsat_indices = []
            for ci in range(m):
                if not is_clause_sat(ci, assignment):
                    if is_clause_falsified(ci, assignment):
                        return False
                    unsat_indices.append(ci)
            
            if not unsat_indices:
                return True

            # Count polarities for each var in unsat clauses
            pos_count = [0] * n
            neg_count = [0] * n
            
            for ci in unsat_indices:
                for var, pol in clauses_v[ci]:
                    if assignment[var] is None:
                        if pol == 1:
                            pos_count[var] += 1
                        else:
                            neg_count[var] += 1
            
            # Assign pure literals
            for var in range(n):
                if assignment[var] is not None:
                    continue
                if pos_count[var] > 0 and neg_count[var] == 0:
                    assignment[var] = True
                    phase[var] = True
                elif neg_count[var] > 0 and pos_count[var] == 0:
                    assignment[var] = False
                    phase[var] = False
            
            # If we assigned anything, we should re-propagate units
            # For simplicity, we return and let the caller re-run propagate_units if needed?
            # Better: integrate or loop.
            # Let's just return True and assume caller calls propagate_units again if necessary.
            # Actually, pure literal elimination can create new units.
            # So we should loop: Pure Lit -> Unit Prop -> Pure Lit...
            # We will handle this in the main dpll_recursive loop.
            
            return True

        def dpll_recursive(assignment, phase, depth, deadline):
            if time.time() > deadline:
                return None
            
            # 1. Pure Literal Elimination
            # We need to check if any pure literals exist.
            # To avoid O(n*m) scan every time, we can do it, since n=50 is small.
            
            # Check for contradiction and units
            if not propagate_units(assignment, phase):
                return None
            
            # Check if solved
            all_sat = True
            for ci in range(m):
                if not is_clause_sat(ci, assignment):
                    if is_clause_falsified(ci, assignment):
                        return None
                    all_sat = False
                    break
            
            if all_sat:
                # Fill unassigned
                for i in range(n):
                    if assignment[i] is None:
                        assignment[i] = phase[i] if phase[i] is not None else False
                return assignment[:]

            # Perform Pure Literal Elimination
            # Identify unsat clauses
            unsat_indices = []
            for ci in range(m):
                if not is_clause_sat(ci, assignment):
                    unsat_indices.append(ci)
            
            pos_count = [0] * n
            neg_count = [0] * n
            for ci in unsat_indices:
                for var, pol in clauses_v[ci]:
                    if assignment[var] is None:
                        if pol == 1:
                            pos_count[var] += 1
                        else:
                            neg_count[var] += 1
            
            # Assign pure literals
            pure_found = False
            for var in range(n):
                if assignment[var] is not None:
                    continue
                if pos_count[var] > 0 and neg_count[var] == 0:
                    assignment[var] = True
                    phase[var] = True
                    pure_found = True
                elif neg_count[var] > 0 and pos_count[var] == 0:
                    assignment[var] = False
                    phase[var] = False
                    pure_found = True
            
            if pure_found:
                # Re-propagate units and check solved again
                if not propagate_units(assignment, phase):
                    return None
                
                all_sat = True
                for ci in range(m):
                    if not is_clause_sat(ci, assignment):
                        if is_clause_falsified(ci, assignment):
                            return None
                        all_sat = False
                        break
                if all_sat:
                    for i in range(n):
                        if assignment[i] is None:
                            assignment[i] = phase[i] if phase[i] is not None else False
                    return assignment[:]
                
                # Recurse immediately as state changed significantly
                return dpll_recursive(assignment, phase, depth + 1, deadline)

            # 2. Branching
            # Heuristic: Pick variable with highest frequency in unsat clauses
            # Among those, pick one with unassigned value.
            
            # Calculate frequency in unsat clauses
            freq = [0] * n
            for ci in unsat_indices:
                for var, pol in clauses_v[ci]:
                    if assignment[var] is None:
                        freq[var] += 1
            
            # Find max freq var that is unassigned
            best_var = -1
            max_f = 0
            for var in range(n):
                if assignment[var] is None and freq[var] > max_f:
                    max_f = freq[var]
                    best_var = var
            
            if best_var == -1:
                # All unassigned vars have freq 0? Then they don't appear in unsat clauses.
                # We can assign them arbitrarily (phase).
                for var in range(n):
                    if assignment[var] is None:
                        assignment[var] = phase[var] if phase[var] is not None else False
                # Final check
                if all(is_clause_sat(ci, assignment) for ci in range(m)):
                    return assignment[:]
                return None

            # Phase Saving: Try phase first
            val1 = phase[best_var] if phase[best_var] is not None else True
            val2 = not val1
            
            # Try val1
            assignment[best_var] = val1
            res = dpll_recursive(assignment, phase, depth + 1, deadline)
            if res is not None:
                return res
            
            # Try val2
            assignment[best_var] = val2
            res = dpll_recursive(assignment, phase, depth + 1, deadline)
            if res is not None:
                return res
            
            # Backtrack
            assignment[best_var] = None
            return None

        # Initial propagation
        if not propagate_units(assignment, phase):
            return None
        
        # Initial check
        all_sat = True
        for ci in range(m):
            if not is_clause_sat(ci, assignment):
                all_sat = False
                break
        
        if all_sat:
            for i in range(n):
                if assignment[i] is None:
                    assignment[i] = False
            return assignment[:]

        # Run DPLL
        dpll_deadline = time.time() + 120.0 # Give DPLL 2 minutes
        result = dpll_recursive(assignment, phase, 0, dpll_deadline)
        
        return result

    # Try DPLL first
    dpll_result = dpll_solve(deadline)
    if dpll_result is not None:
        # Verify
        all_sat = True
        for c in clauses_v:
            sat = False
            for var, pol in c:
                if dpll_result[var] == (pol == 1):
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        if all_sat:
            return dpll_result

    # ============================================================
    # PHASE 2: Optimized WalkSAT
    # ============================================================
    
    # Pre-compute for faster WalkSAT
    # For each clause, store list of (var, pol)
    
    def walksat(n, clauses_v, var_to_clauses, deadline):
        rng = random.Random(42)
        best_assignment = None
        best_unsat_count = float('inf')
        
        # We will track unsat count efficiently
        # Initialize assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        def get_unsat_count(assignment):
            cnt = 0
            for c in clauses_v:
                sat = False
                for var, pol in c:
                    if assignment[var] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    cnt += 1
            return cnt
        
        current_unsat = get_unsat_count(assignment)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            if current_unsat == 0:
                return assignment
        
        max_restarts = 100
        steps_per_restart = 20000
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Reset assignment randomly for each restart? Or continue?
            # Random restart is better for WalkSAT.
            assignment = [rng.random() < 0.5 for _ in range(n)]
            current_unsat = get_unsat_count(assignment)
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                if current_unsat == 0:
                    return assignment
            
            # Track unsatisfied clauses indices
            unsat_indices = []
            for ci, c in enumerate(clauses_v):
                sat = False
                for var, pol in c:
                    if assignment[var] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    unsat_indices.append(ci)
            
            if not unsat_indices:
                return assignment
            
            # WalkSAT Loop
            for step in range(steps_per_restart):
                if time.time() > deadline - 5.0:
                    break
                
                if not unsat_indices:
                    return assignment
                
                # Pick random unsat clause
                ci = rng.choice(unsat_indices)
                clause = clauses_v[ci]
                
                # Score each literal
                best_var = -1
                best_score = -float('inf')
                best_val = False
                
                for var, pol in clause:
                    # If we flip var, how many clauses improve?
                    # Gain: clauses currently unsat that become sat
                    # Loss: clauses currently sat that become unsat
                    gain = 0
                    loss = 0
                    
                    for cj, cpol in var_to_clauses[var]:
                        c_j = clauses_v[cj]
                        
                        # Check current status of c_j
                        cur_sat = False
                        for v, p in c_j:
                            if assignment[v] == (p == 1):
                                cur_sat = True
                                break
                        
                        # Check new status if var is flipped
                        new_val = not assignment[var]
                        new_sat = False
                        for v, p in c_j:
                            if v == var:
                                if new_val == (p == 1):
                                    new_sat = True
                                    break
                            else:
                                if assignment[v] == (p == 1):
                                    new_sat = True
                                    break
                        
                        if cur_sat and not new_sat:
                            loss += 1
                        elif not cur_sat and new_sat:
                            gain += 1
                    
                    score = gain - loss
                    if score > best_score:
                        best_score = score
                        best_var = var
                        best_val = not assignment[var]
                
                # Walk probability
                if rng.random() < 0.5:
                    var, pol = rng.choice(clause)
                    new_val = not assignment[var]
                else:
                    var = best_var
                    new_val = best_val
                
                if var < 0:
                    continue
                
                old_val = assignment[var]
                assignment[var] = new_val
                
                # Update unsat_indices incrementally
                # This is tricky. We need to know which clauses changed status.
                # Clauses affected are var_to_clauses[var]
                new_unsat_indices = []
                for cj in var_to_clauses[var]:
                    # Check if cj is in current unsat_indices or becomes unsat
                    pass # This incremental update is complex to manage with a list of indices.
                
                # For n=50, recalculating unsat_indices is O(M*K) = 1000*3 = 3000 ops.
                # Step count 20000. Total ops 60 million. Python can do ~10-50 million simple ops/sec.
                # 60M ops might take 1-2 seconds. 100 restarts = 100-200 seconds.
                # This is too slow for 290s budget if we do 100 restarts.
                
                # Optimization: Maintain a boolean array `is_unsat` of size m.
                # And a list `unsat_list` of indices where `is_unsat` is True.
                
                pass # See implementation below

        return best_assignment if best_assignment is not None else [False]*n
    
    # Refined WalkSAT with boolean array for unsat tracking
    def walksat_fast(n, clauses_v, var_to_clauses, deadline):
        rng = random.Random(42)
        best_assignment = None
        best_unsat_count = float('inf')
        
        max_restarts = 150
        steps_per_restart = 50000 # Increase steps if faster
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            # Initialize is_unsat
            is_unsat = [False] * m
            unsat_list = []
            for ci, c in enumerate(clauses_v):
                sat = False
                for var, pol in c:
                    if assignment[var] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    is_unsat[ci] = True
                    unsat_list.append(ci)
            
            current_unsat = len(unsat_list)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                if current_unsat == 0:
                    return assignment
            
            for step in range(steps_per_restart):
                if time.time() > deadline - 5.0:
                    break
                
                if not unsat_list:
                    return assignment
                
                ci = rng.choice(unsat_list)
                clause = clauses_v[ci]
                
                # Find best flip
                best_var = -1
                best_score = -float('inf')
                best_val = False
                
                for var, pol in clause:
                    gain = 0
                    loss = 0
                    for cj, cpol in var_to_clauses[var]:
                        # Check cur_sat of cj
                        # Optimization: We can pre-calculate "true_literal_count" for each clause?
                        # No, just check.
                        cur_sat = False
                        for v, p in clauses_v[cj]:
                            if assignment[v] == (p == 1):
                                cur_sat = True
                                break
                        
                        new_val = not assignment[var]
                        new_sat = False
                        for v, p in clauses_v[cj]:
                            if v == var:
                                if new_val == (p == 1):
                                    new_sat = True
                                    break
                            else:
                                if assignment[v] == (p == 1):
                                    new_sat = True
                                    break
                        
                        if cur_sat and not new_sat:
                            loss += 1
                        elif not cur_sat and new_sat:
                            gain += 1
                    
                    score = gain - loss
                    if score > best_score:
                        best_score = score
                        best_var = var
                        best_val = not assignment[var]
                
                if rng.random() < 0.5:
                    var, pol = rng.choice(clause)
                    new_val = not assignment[var]
                else:
                    var = best_var
                    new_val = best_val
                
                if var < 0:
                    continue
                
                old_val = assignment[var]
                assignment[var] = new_val
                
                # Update is_unsat for affected clauses
                for cj, cpol in var_to_clauses[var]:
                    # Determine new satisfaction status
                    is_sat = False
                    for v, p in clauses_v[cj]:
                        if assignment[v] == (p == 1):
                            is_sat = True
                            break
                    
                    if is_unsat[cj] and is_sat:
                        # Was unsat, now sat
                        is_unsat[cj] = False
                        # Remove from unsat_list? 
                        # Removing from list is O(M). 
                        # Instead, we can leave it in list and filter, or use a set?
                        # If we leave it, we might pick a "stale" unsat clause.
                        # When we pick it, we check if it's actually unsat. If not, skip or pick another.
                        # This is "lazy" deletion.
                        pass
                    elif not is_unsat[cj] and not is_sat:
                        # Was sat, now unsat
                        is_unsat[cj] = True
                        unsat_list.append(cj)
                    
                    # If it was unsat and is still unsat, do nothing.
                    # If it was sat and is still sat, do nothing.
                
                # Optional: Clean up unsat_list occasionally to prevent it from growing too large with stale entries
                if step % 100 == 0:
                    # Filter stale entries
                    new_unsat_list = [c for c in unsat_list if is_unsat[c]]
                    unsat_list = new_unsat_list
                    current_unsat = len(unsat_list)
                    if current_unsat < best_unsat_count:
                        best_unsat_count = current_unsat
                        best_assignment = assignment[:]
                    if current_unsat == 0:
                        return assignment

        return best_assignment if best_assignment is not None else [False]*n

    # Run WalkSAT
    ws_deadline = deadline - 5.0
    result = walksat_fast(n, clauses_v, var_to_clauses, ws_deadline)
    
    # Final verification
    if result:
        all_sat = True
        for c in clauses_v:
            sat = False
            for var, pol in c:
                if result[var] == (pol == 1):
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        if all_sat:
            return result
    
    return [False] * n
# EVOLVE-BLOCK-END
