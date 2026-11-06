# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # safety margin before 330s kill
    
    # Preprocess: convert to 0-based, build clause arrays
    m = len(clauses)
    # clause_vars[i] = list of (var_idx, sign) where sign is +1 or -1
    clause_vars = []
    for c in clauses:
        cv = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cv.append((v, s))
        clause_vars.append(cv)
    
    # Build variable-to-clause incidence
    var_to_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for v, s in cv:
            var_to_clauses[v].append(i)
    
    # Track clause satisfaction: for each clause, count of satisfied literals
    # sat_count[i] = number of literals in clause i that are satisfied
    # We also track for each clause which literals are satisfied (for quick check)
    
    best_assignment = None
    best_unsat = float('inf')
    
    rng = random.Random(42)
    
    def get_assignment():
        # Random initial assignment
        return [rng.random() < 0.5 for _ in range(n)]
    
    def compute_sat_counts(assignment):
        """Compute satisfaction count for each clause."""
        counts = np.zeros(m, dtype=np.int32)
        for i, cv in enumerate(clause_vars):
            cnt = 0
            for v, s in cv:
                if assignment[v] == (s == 1):
                    cnt += 1
            counts[i] = cnt
        return counts
    
    def update_clause_counts(assignment, var_idx, old_val, new_val):
        """Update clause satisfaction counts after flipping var_idx."""
        counts = None  # We'll recompute locally
        # This is called in the main loop, so we maintain counts externally
        pass
    
    # Main search loop with adaptive restarts
    max_restarts = 5000
    restart_count = 0
    
    while time.time() < deadline and restart_count < max_restarts:
        restart_count += 1
        
        assignment = get_assignment()
        sat_counts = compute_sat_counts(assignment)
        
        # If already satisfied
        if np.min(sat_counts) > 0:
            return assignment
        
        # Track best
        cur_unsat = int(np.sum(sat_counts == 0))
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        
        # WalkSAT with adaptive parameters
        # Use a phase-based approach: sometimes greedy, sometimes random
        step_budget = 5000
        p_greedy = 0.55  # probability of picking literal that maximizes satisfied clauses
        
        for step in range(step_budget):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_indices = np.where(sat_counts == 0)[0]
            if len(unsat_indices) == 0:
                return assignment
            
            # Track best during walk
            cur_unsat = len(unsat_indices)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            clause_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            clause = clause_vars[clause_idx]
            
            if rng.random() < p_greedy:
                # Greedy: pick literal that maximizes number of newly satisfied clauses
                best_score = -1
                best_var = -1
                best_sign = -1
                for v, s in clause:
                    if assignment[v] == (s == 1):
                        continue  # already satisfied, flipping would unsatisfy
                    # Flipping v would satisfy this literal
                    # Count how many other clauses would become satisfied
                    score = 0
                    for ci in var_to_clauses[v]:
                        if sat_counts[ci] == 1:
                            # Flipping v would satisfy this clause (it has exactly 1 sat literal)
                            # Check if the literal in clause ci for var v is the unsatisfied one
                            for vv, ss in clause_vars[ci]:
                                if vv == v:
                                    if assignment[v] != (ss == 1):
                                        # This literal is currently unsatisfied
                                        # If we flip v, does it satisfy?
                                        if (ss == 1 and not assignment[v]) or (ss == -1 and assignment[v]):
                                            score += 1
                                    break
                    if score > best_score:
                        best_score = score
                        best_var = v
                        best_sign = s
                if best_var >= 0:
                    var_to_flip = best_var
                else:
                    var_to_flip = clause[rng.randint(0, len(clause) - 1)][0]
            else:
                # Random: pick a random literal from the clause
                var_to_flip = clause[rng.randint(0, len(clause) - 1)][0]
            
            # Flip the variable
            old_val = assignment[var_to_flip]
            new_val = not old_val
            assignment[var_to_flip] = new_val
            
            # Update satisfaction counts for all clauses containing var_to_flip
            for ci in var_to_clauses[var_to_flip]:
                # Recompute this clause's satisfaction count
                cnt = 0
                for v, s in clause_vars[ci]:
                    if assignment[v] == (s == 1):
                        cnt += 1
                sat_counts[ci] = cnt
        
        # Check if solved
        if np.min(sat_counts) > 0:
            return assignment
        
        # Adaptive restart: increase step budget slightly
        # p_greedy can also be tuned
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
