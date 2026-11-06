import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Preprocess: convert to 0-based indices and signs
    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        var_list = []
        for l in c:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            var_list.append((v, s))
            var_to_clauses[v].append(i)
        clause_vars.append(var_list)
    
    # For each variable, list of (clause_idx, sign) for quick lookup
    var_clause_sigs = [[] for _ in range(n)]
    for v in range(n):
        for ci in var_to_clauses[v]:
            for (vv, ss) in clause_vars[ci]:
                if vv == v:
                    var_clause_sigs[v].append((ci, ss))
                    break
    
    rng = np.random.RandomState(42)
    
    best_assign = None
    best_unsat = float('inf')
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize random assignment
        assign = rng.randint(0, 2, size=n).astype(np.int8)
        
        # Compute initial clause satisfied literal counts
        clause_sat_count = np.zeros(m, dtype=np.int8)
        for i in range(m):
            cnt = 0
            for (v, s) in clause_vars[i]:
                if s * (2 * int(assign[v]) - 1) > 0:
                    cnt += 1
            clause_sat_count[i] = cnt
        
        unsat_count = int(np.sum(clause_sat_count == 0))
        
        if unsat_count == 0:
            return assign.tolist()
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assign.copy()
        
        # Adaptive walk probability
        walk_prob = 0.3
        no_improve_steps = 0
        last_unsat = unsat_count
        step = 0
        max_steps = 20000
        
        while step < max_steps:
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assign.tolist()
            
            # Select random unsatisfied clause
            unsat_indices = np.where(clause_sat_count == 0)[0]
            if len(unsat_indices) == 0:
                return assign.tolist()
            
            c_idx = int(rng.choice(unsat_indices))
            
            c_vars = clause_vars[c_idx]
            
            # Choose variable to flip
            if rng.random() < walk_prob:
                # Random walk step
                var_idx, sign = c_vars[int(rng.randint(0, len(c_vars)))]
            else:
                # Greedy: pick var in c that minimizes gain
                best_gain = float('inf')
                best_var = None
                
                for (v, s) in c_vars:
                    old_val = int(assign[v])
                    new_val = 1 - old_val
                    old_lit_sat = s * (2 * old_val - 1) > 0
                    new_lit_sat = s * (2 * new_val - 1) > 0
                    
                    gain = 0
                    for (ci, cs) in var_clause_sigs[v]:
                        old_cnt = clause_sat_count[ci]
                        
                        if old_lit_sat and not new_lit_sat:
                            if old_cnt == 1:
                                gain += 1
                        elif not old_lit_sat and new_lit_sat:
                            if old_cnt == 0:
                                gain -= 1
                    
                    if gain < best_gain:
                        best_gain = gain
                        best_var = v
                
                if best_var is not None:
                    var_idx = best_var
                else:
                    var_idx, sign = c_vars[int(rng.randint(0, len(c_vars)))]
            
            # Perform flip
            old_val = int(assign[var_idx])
            assign[var_idx] = 1 - old_val
            new_val = 1 - old_val
            
            # Update clause satisfied counts
            for ci in var_to_clauses[var_idx]:
                # Find the sign of var_idx in this clause
                old_lit_sat = False
                new_lit_sat = False
                for (v, s) in clause_vars[ci]:
                    if v == var_idx:
                        old_lit_sat = s * (2 * old_val - 1) > 0
                        new_lit_sat = s * (2 * new_val - 1) > 0
                        break
                
                old_cnt = clause_sat_count[ci]
                if old_lit_sat and not new_lit_sat:
                    clause_sat_count[ci] = old_cnt - 1
                elif not old_lit_sat and new_lit_sat:
                    clause_sat_count[ci] = old_cnt + 1
                
                # Update unsat count
                if old_cnt == 0 and clause_sat_count[ci] > 0:
                    unsat_count -= 1
                elif old_cnt > 0 and clause_sat_count[ci] == 0:
                    unsat_count += 1
            
            step += 1
            
            # Adaptive walk probability
            if unsat_count < last_unsat:
                last_unsat = unsat_count
                no_improve_steps = 0
                walk_prob = max(0.1, walk_prob * 0.95)
            else:
                no_improve_steps += 1
                if no_improve_steps % 500 == 0:
                    walk_prob = min(0.8, walk_prob * 1.1)
            
            if no_improve_steps > 5000:
                break
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign.copy()
        
        # End of inner loop
    
    # Final polish: try to satisfy remaining unsatisfied clauses
    if best_assign is not None:
        final_assign = best_assign.copy()
        final_unsat = best_unsat
        
        if final_unsat > 0:
            # Compute clause sat counts for final_assign
            final_clause_sat_count = np.zeros(m, dtype=np.int8)
            for i in range(m):
                cnt = 0
                for (v, s) in clause_vars[i]:
                    if s * (2 * int(final_assign[v]) - 1) > 0:
                        cnt += 1
                final_clause_sat_count[i] = cnt
            
            # Try to fix unsatisfied clauses
            for _ in range(200):
                if time.time() > deadline:
                    break
                if final_unsat == 0:
                    break
                
                unsat_indices = np.where(final_clause_sat_count == 0)[0]
                if len(unsat_indices) == 0:
                    break
                
                c_idx = int(unsat_indices[0])
                c_vars = clause_vars[c_idx]
                
                # Try flipping each variable in this clause
                best_gain = float('inf')
                best_var = None
                for (v, s) in c_vars:
                    old_val = int(final_assign[v])
                    new_val = 1 - old_val
                    old_lit_sat = s * (2 * old_val - 1) > 0
                    new_lit_sat = s * (2 * new_val - 1) > 0
                    
                    gain = 0
                    for (ci, cs) in var_clause_sigs[v]:
                        old_cnt = final_clause_sat_count[ci]
                        if old_lit_sat and not new_lit_sat:
                            if old_cnt == 1:
                                gain += 1
                        elif not old_lit_sat and new_lit_sat:
                            if old_cnt == 0:
                                gain -= 1
                    
                    if gain < best_gain:
                        best_gain = gain
                        best_var = v
                
                if best_var is not None and best_gain <= 0:
                    old_val = int(final_assign[best_var])
                    final_assign[best_var] = 1 - old_val
                    new_val = 1 - old_val
                    
                    for ci in var_to_clauses[best_var]:
                        old_lit_sat = False
                        new_lit_sat = False
                        for (v, s) in clause_vars[ci]:
                            if v == best_var:
                                old_lit_sat = s * (2 * old_val - 1) > 0
                                new_lit_sat = s * (2 * new_val - 1) > 0
                                break
                        
                        old_cnt = final_clause_sat_count[ci]
                        if old_lit_sat and not new_lit_sat:
                            final_clause_sat_count[ci] = old_cnt - 1
                        elif not old_lit_sat and new_lit_sat:
                            final_clause_sat_count[ci] = old_cnt + 1
                        
                        if old_cnt == 0 and final_clause_sat_count[ci] > 0:
                            final_unsat -= 1
                        elif old_cnt > 0 and final_clause_sat_count[ci] == 0:
                            final_unsat += 1
                    
                    if final_unsat == 0:
                        break
                else:
                    break
        
        if final_unsat == 0:
            return final_assign.tolist()
    
    if best_assign is not None:
        return best_assign.tolist()
    return [False] * n
