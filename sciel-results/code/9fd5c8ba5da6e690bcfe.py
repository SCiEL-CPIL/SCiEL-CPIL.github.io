# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard
    deadline = time.time() + 325.0
    
    # Convert clauses to 0-based (var_idx, polarity)
    clauses_v = [[(abs(l)-1, 1 if l > 0 else -1) for l in c] for c in clauses]
    m = len(clauses_v)
    
    # Build index: for each variable, list of (clause_idx, polarity)
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_v):
        for var, pol in c:
            var_to_clauses[var].append((ci, pol))
    
    # ============================================================
    # PHASE 1: DPLL with unit propagation
    # ============================================================
    
    def dpll_solver(deadline):
        """Simple DPLL solver. Returns assignment list of bools or None."""
        # Assignment: -1 unassigned, 0 false, 1 true
        assign = [-1] * n
        # clause_state: -1 satisfied, 0 unsatisfied (all false), 1 unassigned remaining
        # Actually, let's just use a recursive function with backtracking.
        
        # For efficiency, we will use a stack-based approach or simple recursion with depth limit.
        # Given N=50, recursion depth is safe.
        
        # Pre-compute clause literals
        # We will maintain a list of active clauses.
        
        # To keep it simple and robust, we implement a standard DPLL.
        
        def propagate(assign):
            """Unit propagation. Modifies assign in place. Returns False if contradiction."""
            changed = True
            while changed:
                changed = False
                for ci in range(m):
                    if len(clauses_v[ci]) == 0:
                        continue
                    # Check if clause is satisfied or unit
                    unassigned = []
                    false_count = 0
                    for var, pol in clauses_v[ci]:
                        if assign[var] == -1:
                            unassigned.append((var, pol))
                        else:
                            if assign[var] == (pol == 1):
                                # Satisfied, skip
                                break
                            else:
                                false_count += 0 # Just checking
                    else:
                        # Loop finished without break (not satisfied)
                        if len(unassigned) == 0:
                            return False # Contradiction
                        elif len(unassigned) == 1:
                            var, pol = unassigned[0]
                            assign[var] = 1 if pol == 1 else 0
                            changed = True
            return True

        def dpll_recursive(assign, depth):
            if time.time() > deadline:
                return None
            
            # Check if all clauses satisfied
            all_sat = True
            for ci in range(m):
                sat = False
                for var, pol in clauses_v[ci]:
                    if assign[var] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    all_sat = False
                    break
            
            if all_sat:
                # Fill unassigned
                for i in range(n):
                    if assign[i] == -1:
                        assign[i] = 0
                return assign[:]

            # Unit Propagation
            if not propagate(assign):
                return None
            
            # Check again after propagation
            all_sat = True
            for ci in range(m):
                sat = False
                for var, pol in clauses_v[ci]:
                    if assign[var] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    all_sat = False
                    break
            
            if all_sat:
                for i in range(n):
                    if assign[i] == -1:
                        assign[i] = 0
                return assign[:]

            # Pure Literal Elimination (Optional, can be expensive)
            # Skip for now to keep DPLL fast, rely on branching.

            # Select variable
            # Heuristic: Variable in most clauses? Or just first unassigned?
            # Let's pick variable with highest degree in unsatisfied clauses.
            var_scores = [0] * n
            for ci in range(m):
                sat = False
                for var, pol in clauses_v[ci]:
                    if assign[var] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    for var, pol in clauses_v[ci]:
                        if assign[var] == -1:
                            var_scores[var] += 1
            
            best_var = -1
            max_score = 0
            for i in range(n):
                if assign[i] == -1 and var_scores[i] > max_score:
                    max_score = var_scores[i]
                    best_var = i
            
            if best_var == -1:
                # No unassigned variables, but not all sat? Contradiction
                return None

            # Branch: Try True
            assign[best_var] = 1
            result = dpll_recursive(assign, depth + 1)
            if result is not None:
                return result
            
            # Branch: Try False
            assign[best_var] = 0
            result = dpll_recursive(assign, depth + 1)
            if result is not None:
                return result
            
            # Backtrack
            assign[best_var] = -1
            return None

        return dpll_recursive(assign, 0)

    # Try DPLL first with a short time limit (e.g., 10s)
    dpll_deadline = time.time() + 10.0
    dpll_result = dpll_solver(dpll_deadline)
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
    
    def walksat(deadline):
        rng = random.Random(42)
        best_assignment = None
        best_unsat_count = float('inf')
        
        # Pre-compute clause literals for speed
        # clauses_v is already 0-based (var, pol)
        
        max_restarts = 1000
        base_steps = 10000
        max_steps = 50000
        
        for restart in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Initialize assignment
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            # Compute initial unsat clauses
            unsat_list = []
            for ci in range(m):
                sat = False
                for var, pol in clauses_v[ci]:
                    if assignment[var] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    unsat_list.append(ci)
            
            current_unsat = len(unsat_list)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                if current_unsat == 0:
                    return assignment
            
            steps = min(base_steps + restart * 100, max_steps)
            
            for step in range(steps):
                if time.time() > deadline - 2.0:
                    break
                
                if not unsat_list:
                    return assignment
                
                # Update best
                if len(unsat_list) < best_unsat_count:
                    best_unsat_count = len(unsat_list)
                    best_assignment = assignment[:]
                
                # Pick random unsat clause
                ci = rng.choice(unsat_list)
                clause = clauses_v[ci]
                
                # Score literals
                best_var = -1
                best_score = float('inf') # Minimize newly unsat
                best_flip = False
                
                for var, pol in clause:
                    new_val = not assignment[var]
                    
                    # Count how many clauses this flip would make unsat
                    # and how many it would satisfy
                    newly_unsat = 0
                    newly_sat = 0
                    
                    for cj, cpol in var_to_clauses[var]:
                        c_j = clauses_v[cj]
                        
                        # Check current state of cj
                        cur_sat = False
                        for v, p in c_j:
                            if assignment[v] == (p == 1):
                                cur_sat = True
                                break
                        
                        # Check new state of cj
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
                            newly_unsat += 1
                        elif not cur_sat and new_sat:
                            newly_sat += 1
                    
                    score = newly_unsat - newly_sat
                    if score < best_score:
                        best_score = score
                        best_var = var
                        best_flip = new_val
                
                # Exploration: random flip with prob 0.25
                if rng.random() < 0.25:
                    var, pol = rng.choice(clause)
                    new_val = not assignment[var]
                else:
                    var = best_var
                    new_val = best_flip
                
                if var == -1:
                    continue
                
                old_val = assignment[var]
                assignment[var] = new_val
                
                # Update unsat_list
                # We need to update the status of all clauses containing var
                # To maintain unsat_list efficiently, we can just rebuild it? No, M is small (200ish).
                # Rebuilding unsat_list is O(M*3) = 600 ops.
                # Updating incrementally is complex with list swaps.
                # Given M is small, rebuilding is safer and likely fast enough.
                # Let's try incremental first, but if it's complex, rebuild.
                # Actually, for M~200, rebuilding is ~600 ops. 50k steps * 600 = 30M ops. ~0.3s.
                # This is acceptable.
                
                # Rebuild unsat_list
                unsat_list = []
                for cj in range(m):
                    sat = False
                    for v, p in clauses_v[cj]:
                        if assignment[v] == (p == 1):
                            sat = True
                            break
                    if not sat:
                        unsat_list.append(cj)
            
            # End of restart steps
            if not unsat_list:
                return assignment
        
        return best_assignment

    return walksat(deadline)
# EVOLVE-BLOCK-END
