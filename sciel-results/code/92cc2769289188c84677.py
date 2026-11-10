import sys
sys.setrecursionlimit(10000)

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    import random
    import time
    
    # Pre-compute variable to clause mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_to_clauses[var].append(i)
            
    def walk_sat(clauses, n, var_to_clauses, seed, max_steps_per_restart=10000, max_restarts=50):
        rng = random.Random(seed)
        
        # Initial random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize clause satisfaction status
        is_sat = [False] * len(clauses)
        unsat_clauses = []
        
        def update_clause(ci):
            sat = False
            for lit in clauses[ci]:
                var = abs(lit) - 1
                val = assignment[var]
                is_true = (lit > 0)
                if val == is_true:
                    sat = True
                    break
            is_sat[ci] = sat
            if not sat:
                if ci not in unsat_clauses:
                    unsat_clauses.append(ci)
            else:
                if ci in unsat_clauses:
                    unsat_clauses.remove(ci)
        
        for i in range(len(clauses)):
            update_clause(i)
            
        if not unsat_clauses:
            return assignment
            
        for restart in range(max_restarts):
            if restart > 0:
                assignment = [rng.random() < 0.5 for _ in range(n)]
                is_sat = [False] * len(clauses)
                unsat_clauses = []
                for i in range(len(clauses)):
                    update_clause(i)
                if not unsat_clauses:
                    return assignment
                    
            for step in range(max_steps_per_restart):
                if not unsat_clauses:
                    return assignment
                
                ci = rng.choice(unsat_clauses)
                clause = clauses[ci]
                
                best_var = None
                best_score = float('inf')
                
                for lit in clause:
                    var = abs(lit) - 1
                    count = 0
                    for cj in var_to_clauses[var]:
                        if is_sat[cj]:
                            count += 1
                            
                    score = count + rng.random() * 0.1
                    
                    if score < best_score:
                        best_score = score
                        best_var = var
                        
                if rng.random() < 0.5:
                    lit = rng.choice(clause)
                    best_var = abs(lit) - 1
                    
                assignment[best_var] = not assignment[best_var]
                
                for cj in var_to_clauses[best_var]:
                    update_clause(cj)
                    
        return None

    start_time = time.time()
    timeout = 250
    
    seed = 0
    while time.time() - start_time < timeout:
        result = walk_sat(clauses, n, var_to_clauses, seed, max_steps_per_restart=5000, max_restarts=20)
        if result is not None:
            return result
        seed += 1
        
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
