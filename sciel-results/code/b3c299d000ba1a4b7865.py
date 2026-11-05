import time
import random

def solve(instance):
    n = instance["n"]
    orig_clauses = instance["clauses"]
    
    # 1. Preprocessing: Unit Propagation
    assignment = [None] * n
    active_clauses = [c[:] for c in orig_clauses]
    
    # Unit Propagation
    changed = True
    while changed:
        changed = False
        for i in range(len(active_clauses)):
            if len(active_clauses[i]) == 0:
                return [random.random() < 0.5 for _ in range(n)]
            
            if len(active_clauses[i]) == 1:
                lit = active_clauses[i][0]
                var = abs(lit) - 1
                val = lit > 0
                
                if assignment[var] is not None:
                    if assignment[var] != val:
                        return [random.random() < 0.5 for _ in range(n)]
                    else:
                        active_clauses[i] = []
                        changed = True
                        continue
                
                assignment[var] = val
                changed = True
                
                for j in range(len(active_clauses)):
                    if len(active_clauses[j]) == 0:
                        continue
                    new_clause = []
                    clause_satisfied = False
                    for l in active_clauses[j]:
                        v = abs(l) - 1
                        if v == var:
                            if (l > 0 and val) or (l < 0 and not val):
                                clause_satisfied = True
                                break
                            else:
                                continue
                        else:
                            new_clause.append(l)
                    
                    if clause_satisfied:
                        active_clauses[j] = []
                    else:
                        active_clauses[j] = new_clause
                break
    
    if all(a is not None for a in assignment):
        for c in orig_clauses:
            sat = False
            for l in c:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                return [random.random() < 0.5 for _ in range(n)]
        return assignment

    # 2. WalkSAT with Incremental Updates and Weighted Heuristic
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = random.random() < 0.5
    
    m = len(orig_clauses)
    clause_sat = [False] * m
    sat_count = [0] * m
    var_to_clauses = [[] for _ in range(n)]
    
    # Precompute literal signs for each variable in each clause for efficiency
    var_literal_in_clause = [{} for _ in range(n)]
    
    for i, c in enumerate(orig_clauses):
        count = 0
        for l in c:
            v = abs(l) - 1
            var_to_clauses[v].append(i)
            var_literal_in_clause[v][i] = l
            if assignment[v] == (l > 0):
                count += 1
        sat_count[i] = count
        clause_sat[i] = (count > 0)
        
    num_unsat = sum(1 for s in clause_sat if not s)
    
    if num_unsat == 0:
        return assignment
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    restart_limit = 1000
    rng = random.Random(42)
    
    best_num_unsat = num_unsat
    best_assignment = assignment[:]
    
    steps = 0
    current_restart_limit = restart_limit
    
    while time.time() < deadline:
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_num_unsat:
            best_num_unsat = num_unsat
            best_assignment = assignment[:]
            
        if steps > current_restart_limit:
            for i in range(n):
                assignment[i] = rng.random() < 0.5
            
            for i, c in enumerate(orig_clauses):
                count = 0
                for l in c:
                    v = abs(l) - 1
                    if assignment[v] == (l > 0):
                        count += 1
                sat_count[i] = count
                clause_sat[i] = (count > 0)
            
            num_unsat = sum(1 for s in clause_sat if not s)
            
            steps = 0
            current_restart_limit = int(current_restart_limit * 1.5)
            if current_restart_limit > 100000:
                current_restart_limit = 100000
            continue
        
        # Pick a random unsatisfied clause
        unsat_clause_idx = rng.randrange(m)
        while clause_sat[unsat_clause_idx]:
            unsat_clause_idx = rng.randrange(m)
        
        clause = orig_clauses[unsat_clause_idx]
        
        # Weighted heuristic: score = w1 * (clauses that become satisfied) - w2 * (clauses that become unsatisfied)
        # Use different weights for random vs heuristic flips
        if rng.random() < 0.3:
            # Random flip
            lit = rng.choice(clause)
            var_idx = abs(lit) - 1
        else:
            # Weighted min-conflict: prefer variables that satisfy more clauses without breaking others
            best_score = -float('inf')
            best_var = None
            for lit in clause:
                var_idx = abs(lit) - 1
                satisfied_gain = 0
                unsatisfied_loss = 0
                
                for c_idx in var_to_clauses[var_idx]:
                    lit_in_c = var_literal_in_clause[var_idx][c_idx]
                    old_count = sat_count[c_idx]
                    
                    # If we flip var_idx, how does this clause change?
                    if assignment[var_idx] == (lit_in_c > 0):
                        # Literal was True, becomes False
                        if old_count == 1:
                            unsatisfied_loss += 1
                    else:
                        # Literal was False, becomes True
                        if old_count == 0:
                            satisfied_gain += 1
                
                # Score: emphasize satisfying more, penalize breaking satisfied clauses
                score = satisfied_gain * 1.0 - unsatisfied_loss * 1.5
                
                if score > best_score:
                    best_score = score
                    best_var = var_idx
            
            if best_var is None:
                best_var = abs(rng.choice(clause)) - 1
            var_idx = best_var
            
        # Flip the variable
        old_val = assignment[var_idx]
        new_val = not old_val
        assignment[var_idx] = new_val
        
        # Update sat_count and clause_sat
        for c_idx in var_to_clauses[var_idx]:
            lit_in_c = var_literal_in_clause[var_idx][c_idx]
            was_sat = clause_sat[c_idx]
            old_count = sat_count[c_idx]
            
            if old_val == (lit_in_c > 0):
                new_count = old_count - 1
            else:
                new_count = old_count + 1
                
            new_sat = (new_count > 0)
            
            if was_sat and not new_sat:
                num_unsat += 1
            elif not was_sat and new_sat:
                num_unsat -= 1
                
            sat_count[c_idx] = new_count
            clause_sat[c_idx] = new_sat
            
        steps += 1
        
    # Final polish: local search with tighter heuristic
    if time.time() < deadline - 10:
        polish_deadline = deadline - 5
        for _ in range(100):  # 100 polish iterations
            if num_unsat == 0:
                return assignment
            if time.time() > polish_deadline:
                break
            
            if num_unsat < best_num_unsat:
                best_num_unsat = num_unsat
                best_assignment = assignment[:]
            
            # Pick an unsatisfied clause
            unsat_clause_idx = rng.randrange(m)
            while clause_sat[unsat_clause_idx]:
                unsat_clause_idx = rng.randrange(m)
            
            clause = orig_clauses[unsat_clause_idx]
            
            # Use stronger heuristic: maximize satisfied_gain - 2*unsatisfied_loss
            best_score = -float('inf')
            best_var = None
            for lit in clause:
                var_idx = abs(lit) - 1
                satisfied_gain = 0
                unsatisfied_loss = 0
                
                for c_idx in var_to_clauses[var_idx]:
                    lit_in_c = var_literal_in_clause[var_idx][c_idx]
                    old_count = sat_count[c_idx]
                    
                    if assignment[var_idx] == (lit_in_c > 0):
                        if old_count == 1:
                            unsatisfied_loss += 1
                    else:
                        if old_count == 0:
                            satisfied_gain += 1
                
                score = satisfied_gain * 1.0 - unsatisfied_loss * 2.0
                
                if score > best_score:
                    best_score = score
                    best_var = var_idx
            
            if best_var is not None:
                old_val = assignment[best_var]
                new_val = not old_val
                assignment[best_var] = new_val
                
                for c_idx in var_to_clauses[best_var]:
                    lit_in_c = var_literal_in_clause[best_var][c_idx]
                    was_sat = clause_sat[c_idx]
                    old_count = sat_count[c_idx]
                    
                    if old_val == (lit_in_c > 0):
                        new_count = old_count - 1
                    else:
                        new_count = old_count + 1
                        
                    new_sat = (new_count > 0)
                    
                    if was_sat and not new_sat:
                        num_unsat += 1
                    elif not was_sat and new_sat:
                        num_unsat -= 1
                        
                    sat_count[c_idx] = new_count
                    clause_sat[c_idx] = new_sat
    
    return best_assignment
