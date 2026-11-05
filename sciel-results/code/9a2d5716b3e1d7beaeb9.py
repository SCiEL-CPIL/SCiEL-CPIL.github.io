import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Precompute variable to clauses mapping
    var_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[var_idx].append((i, sign))
            
    m = len(clauses)
    
    deadline = time.time() + 290.0
    
    # Use a fixed seed for reproducibility in tests? 
    # The prompt doesn't specify, but random restarts help.
    # We will use random.random() which is seeded by system time or random.
    
    while time.time() < deadline:
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        clause_sat_count = [0] * m
        
        # Initialize counts
        for c_idx in range(m):
            count = 0
            for lit in clauses[c_idx]:
                var_idx = abs(lit) - 1
                val = assignment[var_idx]
                if (lit > 0 and val) or (lit < 0 and not val):
                    count += 1
            clause_sat_count[c_idx] = count
            
        if all(c > 0 for c in clause_sat_count):
            return assignment
            
        steps = 0
        # Increase max_steps over time? Or keep fixed?
        # Fixed 50,000 steps per restart is a good balance.
        max_steps = 50000 
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            
            unsat_clauses = [i for i, c in enumerate(clause_sat_count) if c == 0]
            
            if not unsat_clauses:
                return assignment
                
            c_idx = random.choice(unsat_clauses)
            clause = clauses[c_idx]
            
            best_var = -1
            best_gain = -1000
            # Store gains to allow random pick if all negative?
            # Just pick best.
            for lit in clause:
                var_idx = abs(lit) - 1
                current_val = assignment[var_idx]
                
                gain = 0
                for c_idx2, sign in var_clauses[var_idx]:
                    is_sat = (sign == 1 and current_val) or (sign == -1 and not current_val)
                    if is_sat:
                        gain -= 1
                    else:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_var = var_idx
            
            # Heuristic: If best gain is negative, randomize
            if best_gain < 0 and random.random() < 0.5:
                # Pick random var in clause
                best_var = random.choice([abs(l)-1 for l in clause])

            # Flip
            old_val = assignment[best_var]
            new_val = not old_val
            assignment[best_var] = new_val
            
            # Update counts
            for c_idx2, sign in var_clauses[best_var]:
                # Check if literal was satisfied with old_val
                is_sat_old = (sign == 1 and old_val) or (sign == -1 and not old_val)
                if is_sat_old:
                    clause_sat_count[c_idx2] -= 1
                else:
                    clause_sat_count[c_idx2] += 1
                    
    return assignment
