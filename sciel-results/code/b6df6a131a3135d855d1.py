import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 290.0
    
    # Pre-process: Build variable to clause index map
    var_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for lit in c:
            var_idx = abs(lit) - 1
            var_clauses[var_idx].append(i)
            
    assignment = [0] * n
    weights = [0.0] * n
    phase = [0] * n
    
    def get_val(lit):
        v = assignment[abs(lit) - 1]
        if lit > 0:
            return v
        else:
            return 1 - v

    def is_sat(lit):
        return get_val(lit) == 1

    def count_unsat():
        cnt = 0
        for c in clauses:
            sat = False
            for lit in c:
                if is_sat(lit):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    # Initial Random Assignment
    for i in range(n):
        assignment[i] = random.randint(0, 1)
        phase[i] = assignment[i]

    best_unsat = count_unsat()
    best_assignment = assignment[:]
    
    if best_unsat == 0:
        return [bool(x) for x in best_assignment]

    def calc_delta(var_idx, new_val):
        delta = 0
        for c_idx in var_clauses[var_idx]:
            c = clauses[c_idx]
            
            # Find the literal in c corresponding to var_idx
            lit_for_var = 0
            for lit in c:
                if abs(lit) - 1 == var_idx:
                    lit_for_var = lit
                    break
            
            cur_sat_lit = is_sat(lit_for_var)
            
            if lit_for_var > 0:
                new_sat_lit = (new_val == 1)
            else:
                new_sat_lit = (new_val == 0)
                
            # Check if other literals satisfy the clause
            other_sat = False
            for lit in c:
                if abs(lit) - 1 != var_idx:
                    if is_sat(lit):
                        other_sat = True
                        break
            
            if not other_sat:
                if cur_sat_lit and not new_sat_lit:
                    delta -= 1
                elif not cur_sat_lit and new_sat_lit:
                    delta += 1
                    
        return delta

    max_steps_per_restart = 10000
    restarts = 0
    max_restarts = 2000
    
    while time.time() < deadline:
        if restarts >= max_restarts:
            break
            
        steps = 0
        while steps < max_steps_per_restart and time.time() < deadline:
            steps += 1
            
            # Find an unsatisfied clause via random sampling
            unsat_clause_idx = -1
            for _ in range(20):
                idx = random.randint(0, len(clauses) - 1)
                c = clauses[idx]
                sat = False
                for lit in c:
                    if is_sat(lit):
                        sat = True
                        break
                if not sat:
                    unsat_clause_idx = idx
                    break
                    
            if unsat_clause_idx == -1:
                # Likely satisfied or very low density
                cur_unsat = count_unsat()
                if cur_unsat == 0:
                    return [bool(x) for x in assignment]
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                break
            
            c = clauses[unsat_clause_idx]
            
            # Choose a literal to flip
            best_lit = -1
            best_score = 1000000
            
            for lit in c:
                var_idx = abs(lit) - 1
                target_val = 1 if lit > 0 else 0
                d = calc_delta(var_idx, target_val)
                
                # Score: minimize delta, break ties by weight (higher weight = more active = maybe prefer?)
                # Actually, in WalkSAT, we often prefer to flip variables that have been flipped less recently 
                # to encourage exploration, or more recently to exploit. 
                # Let's use: score = d * 1000 - weights[var_idx]
                # We want MIN score.
                
                score = d * 1000 - weights[var_idx]
                
                if score < best_score:
                    best_score = score
                    best_lit = lit
            
            if best_lit == -1:
                best_lit = c[random.randint(0, 2)]
                
            var_idx = abs(best_lit) - 1
            new_val = 1 if best_lit > 0 else 0
            
            phase[var_idx] = new_val
            assignment[var_idx] = new_val
            weights[var_idx] += 1.0
            
            # Periodic check
            if steps % 100 == 0:
                cur_unsat = count_unsat()
                if cur_unsat == 0:
                    return [bool(x) for x in assignment]
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                    
        # End of restart
        for i in range(n):
            weights[i] *= 0.95
        restarts += 1
        
    return [bool(x) for x in best_assignment]
