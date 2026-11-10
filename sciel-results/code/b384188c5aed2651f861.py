import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocess clauses to 0-based and separate signs
    # clause: list of (var_idx, sign) where sign is 1 for positive, -1 for negative
    # var_idx is 0-based
    processed_clauses = []
    for c in clauses:
        pc = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            pc.append((var, sign))
        processed_clauses.append(pc)
        
    # Map variable to list of clause indices it appears in
    var_clauses = [[] for _ in range(n)]
    for i, pc in enumerate(processed_clauses):
        for var, sign in pc:
            var_clauses[var].append(i)

    def is_satisfied(clause_idx, assignment):
        for var, sign in processed_clauses[clause_idx]:
            if assignment[var] == (sign == 1):
                return True
        return False

    def count_satisfied_clauses(assignment):
        count = 0
        for i in range(len(processed_clauses)):
            if is_satisfied(i, assignment):
                count += 1
        return count

    def get_unsat_clauses(assignment):
        unsat = []
        for i in range(len(processed_clauses)):
            if not is_satisfied(i, assignment):
                unsat.append(i)
        return unsat

    def calc_delta(var_idx, assignment):
        """
        Calculate the change in the number of satisfied clauses if we flip var_idx.
        Returns: (num_newly_satisfied, num_newly_unsatisfied)
        """
        newly_sat = 0
        newly_unsat = 0
        for c_idx in var_clauses[var_idx]:
            # Check if clause c_idx is currently satisfied
            is_sat_before = is_satisfied(c_idx, assignment)
            
            # Temporarily flip
            assignment[var_idx] = not assignment[var_idx]
            is_sat_after = is_satisfied(c_idx, assignment)
            
            # Flip back
            assignment[var_idx] = not assignment[var_idx]
            
            if not is_sat_before and is_sat_after:
                newly_sat += 1
            elif is_sat_before and not is_sat_after:
                newly_unsat += 1
                
        return newly_sat - newly_unsat

    rng = random.Random(42) # Fixed seed for reproducibility in evaluation? No, better to use system random or instance-based? 
    # The task says "locally generated", so instance is fixed per call. 
    # Using a fixed seed inside solve might be bad if called multiple times with different instances? 
    # Actually, solve is called once per instance. 
    # But if the evaluator calls solve multiple times on the same instance? Unlikely.
    # Let's use a random seed based on time or just random.SystemRandom() to avoid bias if multiple runs.
    # However, deterministic is often preferred for debugging. 
    # The seed program used random.Random(0). Let's use a different seed or random.SystemRandom().
    # Let's use random.SystemRandom() for true randomness, or just random.Random() which uses system time.
    
    rng = random.Random()
    
    max_time = 290.0
    start_time = time.time()
    
    restart_limit = 1000
    walk_prob = 0.5 # Probability of random flip
    
    best_assignment = [rng.random() < 0.5 for _ in range(n)]
    best_unsat_count = len(processed_clauses) # Worst case
    
    while time.time() - start_time < max_time:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        steps = 0
        
        while steps < restart_limit:
            # Check if solved
            unsat_clauses = get_unsat_clauses(assignment)
            if not unsat_clauses:
                return assignment
            
            # Update best if better
            if len(unsat_clauses) < best_unsat_count:
                best_unsat_count = len(unsat_clauses)
                best_assignment = assignment[:]
            
            # Pick an unsatisfied clause
            c_idx = rng.choice(unsat_clauses)
            clause = processed_clauses[c_idx]
            
            # Calculate delta for each literal in the clause
            # We want to maximize the delta (newly_sat - newly_unsat)
            # If random, just pick random literal
            
            if rng.random() < walk_prob:
                # Random flip
                var, sign = rng.choice(clause)
                assignment[var] = not assignment[var]
            else:
                # Greedy flip
                best_var = -1
                best_delta = -1000000
                for var, sign in clause:
                    delta = calc_delta(var, assignment)
                    if delta > best_delta:
                        best_delta = delta
                        best_var = var
                if best_var != -1:
                    assignment[best_var] = not assignment[best_var]
                else:
                    # Fallback to random if something went wrong
                    var, sign = rng.choice(clause)
                    assignment[var] = not assignment[var]
                    
            steps += 1
            
            # Check time periodically
            if steps % 100 == 0 and time.time() - start_time > max_time:
                break
        
        # Increase restart limit
        restart_limit = int(restart_limit * 1.5)
        
    return best_assignment
