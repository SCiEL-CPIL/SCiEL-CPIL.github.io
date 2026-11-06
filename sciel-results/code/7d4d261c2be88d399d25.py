import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n

    # Preprocessing: Build occurrence lists
    var_occ = [[] for _ in range(n)]
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            var_occ[var_idx].append((c_idx, lit))
            
    start_time = time.time()
    deadline = start_time + 320.0
    
    rng = random.Random(0)
    
    def init_state(assignment):
        clause_sat_count = [0] * m
        unsat_set = set()
        for c_idx, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                var_idx = abs(lit) - 1
                if assignment[var_idx] == (lit > 0):
                    count += 1
            clause_sat_count[c_idx] = count
            if count == 0:
                unsat_set.add(c_idx)
        return clause_sat_count, unsat_set

    assignment = [rng.random() < 0.5 for _ in range(n)]
    clause_sat_count, unsat_set = init_state(assignment)
    
    if not unsat_set:
        return assignment
        
    best_assignment = assignment[:]
    best_unsat_count = len(unsat_set)
    
    noise_prob = 0.15
    restart_threshold = 2000
    steps_since_improvement = 0
    
    while time.time() < deadline:
        if not unsat_set:
            return assignment
            
        unsat_list = list(unsat_set)
        if not unsat_list:
            return assignment
            
        c_idx = rng.choice(unsat_list)
        clause = clauses[c_idx]
        
        best_var = -1
        best_cost = float('inf')
        
        for lit in clause:
            v = abs(lit) - 1
            cost = 0
            for c_i, l in var_occ[v]:
                is_sat_before = (assignment[v] == (l > 0))
                is_sat_after = (not assignment[v] == (l > 0))
                
                if is_sat_before and not is_sat_after:
                    cost += 1
                elif not is_sat_before and is_sat_after:
                    cost -= 1
                    
            if cost < best_cost:
                best_cost = cost
                best_var = v
            elif cost == best_cost and rng.random() < 0.5:
                best_var = v
        
        if rng.random() < noise_prob:
            lit = rng.choice(clause)
            v = abs(lit) - 1
        else:
            v = best_var
            
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val
        
        for c_i, l in var_occ[v]:
            old_sat = clause_sat_count[c_i]
            
            if old_val == (l > 0):
                clause_sat_count[c_i] = old_sat - 1
                if old_sat - 1 == 0:
                    unsat_set.add(c_i)
            else:
                clause_sat_count[c_i] = old_sat + 1
                if old_sat + 1 == 1:
                    unsat_set.remove(c_i)
                    
        if not unsat_set:
            return assignment
            
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
            
        if steps_since_improvement > restart_threshold:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            clause_sat_count, unsat_set = init_state(assignment)
            steps_since_improvement = 0
            restart_threshold = int(restart_threshold * 1.5)
            
            if not unsat_set:
                return assignment

    return best_assignment
