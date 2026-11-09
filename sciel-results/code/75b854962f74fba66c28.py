import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocess
    var_clauses = [[] for _ in range(n)]
    clause_vars = []
    clause_signs = []
    
    for i, clause in enumerate(clauses):
        vars_in_clause = []
        signs_in_clause = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            vars_in_clause.append(v)
            signs_in_clause.append(s)
            var_clauses[v].append(i)
        clause_vars.append(vars_in_clause)
        clause_signs.append(signs_in_clause)
        
    num_clauses = len(clauses)
    
    def is_satisfied(c_idx):
        vars_c = clause_vars[c_idx]
        signs_c = clause_signs[c_idx]
        for v, s in zip(vars_c, signs_c):
            if assignment[v] == (s == 1):
                return True
        return False

    start_time = time.time()
    deadline = start_time + 290.0
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    noise = 0.5
    max_steps_per_restart = 2000
    
    for restart in range(10000):
        if time.time() > deadline:
            break
            
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        sat = [False] * num_clauses
        unsat_list = []
        unsat_set = set()
        
        for i in range(num_clauses):
            if is_satisfied(i):
                sat[i] = True
            else:
                sat[i] = False
                unsat_list.append(i)
                unsat_set.add(i)
                
        if not unsat_list:
            return assignment
            
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                if len(unsat_list) < best_unsat_count:
                    best_unsat_count = len(unsat_list)
                    best_assignment = assignment[:]
                break
            
            if not unsat_list:
                return assignment
                
            c_idx = random.choice(unsat_list)
            
            vars_c = clause_vars[c_idx]
            signs_c = clause_signs[c_idx]
            
            min_unsat = float('inf')
            candidates = []
            
            for v in vars_c:
                unsat_count = 0
                for j, v2 in enumerate(vars_c):
                    s2 = signs_c[j]
                    if v2 == v:
                        new_val = not assignment[v]
                        if not (new_val == (s2 == 1)):
                            unsat_count += 1
                    else:
                        if not (assignment[v2] == (s2 == 1)):
                            unsat_count += 1
                            
                if unsat_count < min_unsat:
                    min_unsat = unsat_count
                    candidates = [v]
                elif unsat_count == min_unsat:
                    candidates.append(v)
                    
            if random.random() < noise:
                v_to_flip = random.choice(vars_c)
            else:
                v_to_flip = random.choice(candidates)
                
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            for c_idx_2 in var_clauses[v_to_flip]:
                new_sat = is_satisfied(c_idx_2)
                if new_sat != sat[c_idx_2]:
                    sat[c_idx_2] = new_sat
                    if new_sat:
                        unsat_set.discard(c_idx_2)
                        # Remove from list
                        # O(N) operation
                        if c_idx_2 in unsat_list:
                            unsat_list.remove(c_idx_2)
                    else:
                        unsat_set.add(c_idx_2)
                        unsat_list.append(c_idx_2)
                        
        if len(unsat_list) < best_unsat_count:
            best_unsat_count = len(unsat_list)
            best_assignment = assignment[:]
            
    if best_assignment:
        return best_assignment
    return assignment
