import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: 300s limit, use 290s for safety
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Best assignment found so far (fewest unsatisfied clauses)
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Use a deterministic seed for reproducibility if needed, 
    # but random restarts benefit from variety. 
    # The prompt says "locally generated", so instances might vary.
    # Using a fixed seed for the RNG is standard for these tasks to ensure consistency 
    # unless "random" behavior is desired. The seed program used seed 0.
    # I will use a seed derived from the instance or just a fixed seed for the solver logic.
    # Let's stick to a fixed seed for the solver to be deterministic per instance, 
    # but the restarts will use the same RNG stream.
    rng = random.Random(42)
    
    # Parameters
    # For 3-SAT with 50 vars, 1000-5000 steps per restart is usually enough.
    # Let's use 2000 steps per restart.
    steps_per_restart = 2000
    noise = 0.5  # Probability of random flip vs min-conflicts flip
    
    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for clause in clauses:
            sat = False
            for l in clause:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def get_unsat_clauses(assignment):
        """Get list of unsatisfied clauses."""
        unsat = []
        for clause in clauses:
            sat = False
            for l in clause:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                unsat.append(clause)
        return unsat

    while time.time() < deadline:
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Check if initially satisfied
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Get unsatisfied clauses
            # Optimization: We can track unsat clauses, but for N=50, 
            # recomputing is fast enough.
            unsat_clauses = get_unsat_clauses(assignment)
            
            if not unsat_clauses:
                return assignment
            
            current_unsat_count = len(unsat_clauses)
            if current_unsat_count < best_unsat_count:
                best_unsat_count = current_unsat_count
                best_assignment = assignment[:]

            # Pick a random unsatisfied clause
            clause = rng.choice(unsat_clauses)
            
            # Decide flip strategy
            if rng.random() < noise:
                # Random flip
                literal = rng.choice(clause)
                var_idx = abs(literal) - 1
                assignment[var_idx] = not assignment[var_idx]
            else:
                # Min-conflicts flip: flip variable that satisfies most other unsat clauses
                # Actually, standard WalkSAT flips the variable in the clause that 
                # results in the fewest *new* unsatisfied clauses or satisfies 
                # the most currently unsatisfied clauses.
                # Let's use the heuristic: flip the variable in the clause that 
                # satisfies the maximum number of other unsatisfied clauses.
                
                best_var_idx = -1
                best_score = -1
                
                # Precompute unsat clauses for scoring (already have unsat_clauses)
                # We need to check impact on other unsat clauses.
                # For each variable in the chosen clause, count how many 
                # *other* unsat clauses would become satisfied if we flip it.
                
                for literal in clause:
                    var_idx = abs(literal) - 1
                    # Simulate flip
                    new_val = not assignment[var_idx]
                    
                    score = 0
                    # Check all unsat clauses (excluding the current one? 
                    # Usually we count how many unsat clauses become sat).
                    # If we flip var_idx, a clause is satisfied if it contains 
                    # the literal (var_idx, new_val) or (-var_idx, new_val).
                    # Specifically, if clause contains l, and assignment[abs(l)-1] becomes (l>0).
                    
                    for c in unsat_clauses:
                        if c is clause:
                            continue
                        # Check if flipping var_idx satisfies c
                        sat = False
                        for l in c:
                            idx = abs(l) - 1
                            if idx == var_idx:
                                if new_val == (l > 0):
                                    sat = True
                                    break
                            else:
                                if assignment[idx] == (l > 0):
                                    sat = True
                                    break
                        if sat:
                            score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_var_idx = var_idx
                
                if best_var_idx != -1:
                    assignment[best_var_idx] = not assignment[best_var_idx]
                else:
                    # Fallback to random if no variable improves
                    literal = rng.choice(clause)
                    var_idx = abs(literal) - 1
                    assignment[var_idx] = not assignment[var_idx]

    # If we run out of time, return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: return random assignment
        return [rng.random() < 0.5 for _ in range(n)]
