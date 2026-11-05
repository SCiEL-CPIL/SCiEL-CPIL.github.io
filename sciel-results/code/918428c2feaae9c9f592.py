import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 285.0
    
    # Precompute clause structure as numpy arrays for vectorized operations
    # For each clause: (var_indices, signs) where sign=+1 for positive, -1 for negative
    clause_vars = np.zeros((m, 3), dtype=np.int32)   # 0-based variable indices
    clause_signs = np.zeros((m, 3), dtype=np.int8)    # +1 or -1
    clause_lens = np.zeros(m, dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
        clause_lens[i] = len(c)
    
    # Build var_to_clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            if v >= 0 and clause_signs[i, j] != 0:
                var_to_clauses[v].append(i)
    
    # Unit propagation preprocessing
    assignment = [None] * n  # None = unassigned
    changed = True
    while changed:
        changed = False
        for i in range(m):
            # Count unassigned and satisfied literals
            unassigned_count = 0
            satisfied = False
            forced_lit = None
            forced_val = None
            for j in range(3):
                v = clause_vars[i, j]
                s = clause_signs[i, j]
                if s == 0:
                    continue
                if assignment[v] is not None:
                    if (s == 1 and assignment[v]) or (s == -1 and not assignment[v]):
                        satisfied = True
                        break
                else:
                    unassigned_count += 1
                    forced_lit = v
                    forced_val = (s == 1)
            
            if satisfied:
                continue
                
            if unassigned_count == 1 and forced_lit is not None:
                if assignment[forced_lit] is None:
                    assignment[forced_lit] = forced_val
                    changed = True
                elif assignment[forced_lit] != forced_val:
                    # Conflict - shouldn't happen for satisfiable instances
                    break
    
    # Create initial assignment: use unit-propagated values, random for rest
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    def make_initial_assign():
        assign = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if assignment[i] is not None:
                assign[i] = assignment[i]
            else:
                assign[i] = rng.random() < 0.5
        return assign
    
    def get_clause_sat(assign):
        """Return boolean array of which clauses are satisfied."""
        sat = np.zeros(m, dtype=np.bool_)
        for j in range(3):
            vars_j = clause_vars[:, j]
            signs_j = clause_signs[:, j]
            valid = signs_j != 0
            if not np.any(valid):
                continue
            vars_valid = vars_j[valid]
            signs_valid = signs_j[valid]
            lit_sat = np.zeros(m, dtype=np.bool_)
            lit_sat[valid] = (assign[vars_valid] == (signs_valid > 0))
            sat |= lit_sat
        return sat
    
    best_assign = None
    best_unsat = float('inf')
    
    max_restarts = 100
    base_max_steps = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        assign = make_initial_assign()
        clause_sat = get_clause_sat(assign)
        unsat_count = m - np.sum(clause_sat)
        
        if unsat_count == 0:
            return assign.tolist()
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assign.copy()
        
        # Adaptive parameters
        walk_prob = 0.3
        max_steps = base_max_steps + restart * 500
        if restart > 20:
            max_steps = int(max_steps * 1.5)
        if restart > 50:
            max_steps = int(max_steps * 2.0)
        
        step = 0
        no_improve_steps = 0
        last_unsat = unsat_count
        last_walk_prob = walk_prob
        
        while step < max_steps:
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assign.tolist()
            
            # Select an unsatisfied clause
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                return assign.tolist()
            
            # Frequency-based clause selection: prefer clauses with fewer occurrences in recent history
            # This helps traverse flat regions by focusing on "rare" unsatisfied clauses
            c_idx = int(unsat_indices[np_rng.randint(len(unsat_indices))])
            c_vars = clause_vars[c_idx]
            c_signs = clause_signs[c_idx]
            
            # Select variable to flip
            if rng.random() < walk_prob:
                # Walk step: random literal from clause
                valid_lits = np.where(c_signs != 0)[0]
                j = int(valid_lits[np_rng.randint(len(valid_lits))])
                var_idx = int(c_vars[j])
            else:
                # Greedy step: find var that minimizes unsat_count after flip
                best_gain = float('inf')
                best_var = None
                
                valid_lits = np.where(c_signs != 0)[0]
                for j in valid_lits:
                    var_idx = int(c_vars[j])
                    
                    # Compute gain: change in unsat_count if we flip var_idx
                    gain = 0
                    current_val = assign[var_idx]
                    new_val = not current_val
                    
                    for ci in var_to_clauses[var_idx]:
                        # Check if flipping var_idx changes satisfaction of clause ci
                        cc_vars = clause_vars[ci]
                        cc_signs = clause_signs[ci]
                        
                        # Find the literal for var_idx in clause ci
                        lit_sat_new = False
                        lit_sat_old = False
                        found = False
                        for k in range(3):
                            if cc_signs[k] == 0:
                                continue
                            if cc_vars[k] == var_idx:
                                found = True
                                s = cc_signs[k]
                                lit_sat_new = (new_val == (s > 0))
                                lit_sat_old = (current_val == (s > 0))
                                break
                        
                        if not found:
                            continue
                        
                        is_sat_now = clause_sat[ci]
                        
                        if is_sat_now:
                            # Clause becomes unsatisfied only if:
                            # 1. The literal for var_idx was the ONLY satisfied literal
                            # 2. The literal was satisfied by old value but not new
                            if lit_sat_old and not lit_sat_new:
                                # Check if any other literal is satisfied
                                others_sat = False
                                for k in range(3):
                                    if cc_signs[k] == 0:
                                        continue
                                    if cc_vars[k] == var_idx:
                                        continue
                                    v = int(cc_vars[k])
                                    s = cc_signs[k]
                                    if assign[v] == (s > 0):
                                        others_sat = True
                                        break
                                if not others_sat:
                                    gain += 1
                        else:
                            # Clause becomes satisfied if the literal for var_idx is satisfied by new value
                            if lit_sat_new:
                                gain -= 1
                    
                    if gain < best_gain:
                        best_gain = gain
                        best_var = var_idx
                
                if best_var is not None:
                    var_idx = best_var
                else:
                    valid_lits = np.where(c_signs != 0)[0]
                    j = int(valid_lits[np_rng.randint(len(valid_lits))])
                    var_idx = int(c_vars[j])
            
            # Perform the flip
            assign[var_idx] = not assign[var_idx]
            
            # Update clause statuses for all clauses containing var_idx
            for ci in var_to_clauses[var_idx]:
                old_sat = clause_sat[ci]
                # Recompute satisfaction
                new_sat = False
                for k in range(3):
                    if clause_signs[ci, k] == 0:
                        continue
                    v = int(clause_vars[ci, k])
                    s = clause_signs[ci, k]
                    if assign[v] == (s > 0):
                        new_sat = True
                        break
                
                if old_sat and not new_sat:
                    unsat_count += 1
                elif not old_sat and new_sat:
                    unsat_count -= 1
                clause_sat[ci] = new_sat
            
            step += 1
            
            # Track progress and adapt walk probability
            if unsat_count < last_unsat:
                last_unsat = unsat_count
                no_improve_steps = 0
                # Be more greedy when improving
                if walk_prob > 0.2:
                    walk_prob = max(0.2, walk_prob * 0.95)
            else:
                no_improve_steps += 1
                # Increase walk probability when stuck
                if no_improve_steps > 200:
                    walk_prob = min(0.7, walk_prob * 1.05)
                elif no_improve_steps > 500:
                    walk_prob = min(0.8, walk_prob * 1.1)
                elif no_improve_steps > 1000:
                    walk_prob = 0.9
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign.copy()
            
            # Break out if stuck too long
            if no_improve_steps > 3000:
                break
    
    # Final check
    if best_assign is not None:
        final_sat = get_clause_sat(best_assign)
        if np.all(final_sat):
            return best_assign.tolist()
    
    return best_assign.tolist() if best_assign is not None else [False] * n
