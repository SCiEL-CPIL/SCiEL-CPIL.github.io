import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    start_time = time.time()
    time_limit = 2.0
    
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    processed_clauses = []
    for c in clauses:
        pc = []
        for l in c:
            var = abs(l) - 1
            is_pos = l > 0
            pc.append((var, is_pos))
        processed_clauses.append(pc)
        
    var_freq = [0] * n
    for pc in processed_clauses:
        for var, _ in pc:
            var_freq[var] += 1
            
    def is_satisfied(clause_idx):
        clause = processed_clauses[clause_idx]
        for var, is_pos in clause:
            if assignment[var] == is_pos:
                return True
        return False
        
    num_clauses = len(processed_clauses)
    
    # Check if solved
    # We can just loop and try to find an unsat clause.
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        # Find an unsatisfied clause
        # Pick a random clause index
        # If it's satisfied, try again? 
        # Or loop through all to find one?
        # Looping through all is O(M). Picking random is O(1) but might fail.
        # If most clauses are satisfied, random picking is bad.
        # If few are satisfied, random picking is good.
        # Let's loop through all to find the first unsat one.
        # M is small (~200). This is fast.
        
        unsat_idx = -1
        for i in range(num_clauses):
            if not is_satisfied(i):
                unsat_idx = i
                break
                
        if unsat_idx == -1:
            # All satisfied
            break
            
        clause = processed_clauses[unsat_idx]
        
        # Choose literal
        if rng.random() < 0.5:
            best_var = -1
            min_freq = float('inf')
            for var, _ in clause:
                if var_freq[var] < min_freq:
                    min_freq = var_freq[var]
                    best_var = var
            target_var = best_var
        else:
            target_var, _ = rng.choice(clause)
            
        assignment[target_var] = not assignment[target_var]
        
    return assignment
