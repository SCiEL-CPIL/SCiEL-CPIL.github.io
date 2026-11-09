import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Preprocessing
    # var_clauses[v] = list of (clause_idx, literal_in_clause)
    var_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            var_clauses[v].append((i, lit))
            
    best_assignment = None
    best_unsat_count = m + 1
    
    while time.time() < deadline:
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        false_count = [0] * m
        unsat_clauses = set()
        
        for i, clause in enumerate(clauses):
            fc = 0
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] != (lit > 0):
                    fc += 1
            false_count[i] = fc
            if fc > 0:
                unsat_clauses.add(i)
                
        if not unsat_clauses:
            return assignment
            
        if len(unsat_clauses) < best_unsat_count:
            best_unsat_count = len(unsat_clauses)
            best_assignment = assignment[:]
            
        no_improve_steps = 0
        last_improve_unsat = len(unsat_clauses)
        
        while time.time() < deadline:
            if not unsat_clauses:
                return assignment
                
            cur_unsat = len(unsat_clauses)
            if cur_unsat < best_unsat_count:
                best_unsat_count = cur_unsat
                best_assignment = assignment[:]
                last_improve_unsat = cur_unsat
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                if no_improve_steps > 2000:
                    break
                    
            # Select unsatisfied clause
            # Random choice from unsat_clauses
            clause_idx = random.choice(list(unsat_clauses))
            clause = clauses[clause_idx]
            
            # Calculate gain for each literal in clause
            gains = []
            for lit in clause:
                v = abs(lit) - 1
                gain = 0
                for c_idx, lit_in_c in var_clauses[v]:
                    if c_idx in unsat_clauses:
                        # Check if lit_in_c is currently false
                        if assignment[v] != (lit_in_c > 0):
                            gain += 1
                    else:
                        # Check if flipping v makes this clause unsat
                        # This happens if false_count[c_idx] == len(clauses[c_idx]) - 1
                        # AND lit_in_c is currently true
                        if false_count[c_idx] == len(clauses[c_idx]) - 1:
                            if assignment[v] == (lit_in_c > 0):
                                gain -= 1
                gains.append(gain)
                
            # Pick best literal
            max_gain = max(gains)
            # If random tie-breaking is better?
            # For Min-Conflicts, usually pick best. If tie, random.
            best_lits = [clause[i] for i, g in enumerate(gains) if g == max_gain]
            chosen_lit = random.choice(best_lits)
            
            # Flip
            v = abs(chosen_lit) - 1
            assignment[v] = not assignment[v]
            
            # Update
            for c_idx, lit_in_c in var_clauses[v]:
                # Determine change in false_count
                # Current assignment[v] is NEW
                # Old assignment[v] was NOT NEW
                
                is_true_now = assignment[v] == (lit_in_c > 0)
                is_true_before = not is_true_now
                
                if is_true_before and not is_true_now:
                    # True -> False
                    false_count[c_idx] += 1
                    if false_count[c_idx] == 1: # Was 0, now 1
                        unsat_clauses.add(c_idx)
                elif not is_true_before and is_true_now:
                    # False -> True
                    false_count[c_idx] -= 1
                    if false_count[c_idx] == 0:
                        unsat_clauses.remove(c_idx)
                        
    if best_assignment is not None:
        return best_assignment
    return [False] * n
