import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert to 0-based internal representation
    internal_clauses = []
    for clause in clauses:
        ic = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            ic.append((var, sign))
        internal_clauses.append(ic)
    
    m = len(internal_clauses)
    if m == 0:
        return [True] * n
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 300s limit
    
    # Preprocessing: Pure Literal Elimination and Unit Propagation
    assignment = [None] * n
    active_clauses = list(range(m))
    
    var_pos = [set() for _ in range(n)]
    var_neg = [set() for _ in range(n)]
    for ci, clause in enumerate(internal_clauses):
        for var, sign in clause:
            if sign > 0:
                var_pos[var].add(ci)
            else:
                var_neg[var].add(ci)
    
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
            
        # Pure literal elimination
        for var in range(n):
            if assignment[var] is not None:
                continue
            if len(var_pos[var]) == 0 and len(var_neg[var]) > 0:
                assignment[var] = False
                changed = True
            elif len(var_neg[var]) == 0 and len(var_pos[var]) > 0:
                assignment[var] = True
                changed = True
        
        # Unit propagation
        for ci in active_clauses:
            if ci >= m:
                continue
            unassigned = [(var, sign) for var, sign in internal_clauses[ci] 
                         if assignment[var] is None]
            if len(unassigned) == 1:
                var, sign = unassigned[0]
                if assignment[var] is None:
                    assignment[var] = (sign > 0)
                    changed = True
            elif len(unassigned) == 0:
                # Check if satisfied
                is_sat = any(assignment[var] == (sign > 0) 
                            for var, sign in internal_clauses[ci] 
                            if assignment[var] is not None)
                if not is_sat:
                    pass

    # Build reduced instance
    unassigned_vars = [i for i in range(n) if assignment[i] is None]
    num_unassigned = len(unassigned_vars)
    var_to_idx = {v: i for i, v in enumerate(unassigned_vars)}
    
    reduced_clauses = []
    for ci in range(m):
        rc = []
        is_sat = False
        for var, sign in internal_clauses[ci]:
            if assignment[var] is not None:
                if assignment[var] == (sign > 0):
                    is_sat = True
                    break
            else:
                rc.append((var_to_idx[var], sign))
        if not is_sat:
            reduced_clauses.append(rc)
            
    if num_unassigned == 0:
        for clause in internal_clauses:
            if not any(assignment[var] == (sign > 0) for var, sign in clause):
                return [assignment[i] if assignment[i] is not None else True for i in range(n)]
        return [assignment[i] if assignment[i] is not None else True for i in range(n)]
        
    if len(reduced_clauses) == 0:
        for v in unassigned_vars:
            assignment[v] = True
        return assignment

    # --- Efficient WalkSAT ---
    num_vars = num_unassigned
    num_clauses = len(reduced_clauses)
    
    # Precompute structures for speed
    # clause_vars[c]: list of var indices
    # clause_signs[c]: list of signs
    clause_vars = [None] * num_clauses
    clause_signs = [None] * num_clauses
    var_clauses = [[] for _ in range(num_vars)] # var_clauses[v] = list of (clause_idx, sign)
    
    for ci, clause in enumerate(reduced_clauses):
        vars_in_c = []
        signs_in_c = []
        for v, sign in clause:
            vars_in_c.append(v)
            signs_in_c.append(sign)
            var_clauses[v].append((ci, sign))
        clause_vars[ci] = vars_in_c
        clause_signs[ci] = signs_in_c
        
    rng = random.Random(42)
    
    best_state = None
    best_unsat_count = float('inf')
    
    max_restarts = 5000
    restart_count = 0
    
    while time.time() < deadline and restart_count < max_restarts:
        restart_count += 1
        
        # Random initialization
        state = [rng.random() < 0.5 for _ in range(num_vars)]
        
        # Calculate initial sat_counts and unsat_clauses
        sat_counts = [0] * num_clauses
        unsat_clauses = []
        for ci in range(num_clauses):
            count = 0
            vars_c = clause_vars[ci]
            signs_c = clause_signs[ci]
            for i in range(len(vars_c)):
                v = vars_c[i]
                sign = signs_c[i]
                if state[v] == (sign > 0):
                    count += 1
            sat_counts[ci] = count
            if count == 0:
                unsat_clauses.append(ci)
                
        current_unsat = len(unsat_clauses)
        
        if current_unsat == 0:
            for i, v in enumerate(unassigned_vars):
                assignment[v] = state[i]
            return assignment
            
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_state = state[:]
            
        max_steps = 10000
        step = 0
        
        while unsat_clauses and step < max_steps:
            if time.time() > deadline:
                break
                
            # Select an unsatisfied clause
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            
            # Determine flip variable
            p_noise = 0.35
            
            if rng.random() < p_noise:
                # Random flip
                v = clause_vars[ci][rng.randrange(len(clause_vars[ci]))]
            else:
                # Min-Conflicts
                best_v = -1
                best_score = float('inf')
                
                for i in range(len(clause_vars[ci])):
                    v = clause_vars[ci][i]
                    score = 0
                    # Iterate over clauses containing v
                    for cj, sign_v_in_cj in var_clauses[v]:
                        if cj == ci:
                            continue
                        # Check if cj is currently satisfied
                        if sat_counts[cj] > 0:
                            # If satisfied, check if v is the only satisfying literal
                            # We need to check if any other literal in cj is satisfied
                            # This is expensive. 
                            # Optimization: 
                            # If sat_counts[cj] > 1, v is not the only one, so flipping v doesn't make it unsat.
                            if sat_counts[cj] == 1:
                                # v might be the only one. Check if v is currently satisfying cj.
                                if state[v] == (sign_v_in_cj > 0):
                                    # v is satisfying cj, and it's the only one.
                                    # Flipping v will make cj unsatisfied.
                                    score += 1
                        else:
                            # cj is unsatisfied.
                            # If v is currently falsifying cj, flipping v will satisfy it.
                            if state[v] != (sign_v_in_cj > 0):
                                score -= 1
                                
                    if score < best_score:
                        best_score = score
                        best_v = v
                        
                v = best_v
                
            # Flip variable v
            state[v] = not state[v]
            
            # Update sat_counts and unsat_clauses
            for cj, sign_v_in_cj in var_clauses[v]:
                # Determine if v was satisfying or falsifying cj BEFORE flip
                # state[v] is the NEW value.
                # OLD value was not state[v].
                old_satisfies = (not state[v]) == (sign_v_in_cj > 0)
                
                if old_satisfies:
                    # v was satisfying, now it is not.
                    sat_counts[cj] -= 1
                    if sat_counts[cj] == 0:
                        unsat_clauses.append(cj)
                else:
                    # v was falsifying, now it is satisfying.
                    sat_counts[cj] += 1
                    if sat_counts[cj] == 1:
                        # Became satisfied. 
                        # We don't remove from unsat_clauses immediately to avoid O(N) cost.
                        # We will clean up periodically.
                        pass
                        
            # Periodic cleanup
            if step % 50 == 0:
                # Filter unsat_clauses
                # Keep only those with sat_counts[cj] == 0
                new_unsat = [cj for cj in unsat_clauses if sat_counts[cj] == 0]
                unsat_clauses = new_unsat
                current_unsat = len(unsat_clauses)
                
                if current_unsat == 0:
                    break
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_state = state[:]
                    
            step += 1
            
        if time.time() > deadline:
            break
            
    if best_state:
        for i, v in enumerate(unassigned_vars):
            assignment[v] = best_state[i]
    else:
        for i, v in enumerate(unassigned_vars):
            assignment[v] = True
            
    return assignment
