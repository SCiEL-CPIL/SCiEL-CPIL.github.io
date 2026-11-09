"""WalkSAT with optimized data structures and adaptive parameters."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 330s wall-clock limit, use 295s for safety
    start_time = time.time()
    deadline = start_time + 295.0
    
    # Preprocess: convert clauses to 0-based variable indices and signs
    # clause_vars[c] = array of (var_idx, sign) where sign is +1 or -1
    clause_vars = np.empty((m, 3), dtype=np.int8)
    for c, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_vars[c, j] = var_idx * 2 + (1 if sign > 0 else 0)
    
    # Extract variable indices and signs separately for faster access
    clause_var_idx = (clause_vars // 2).astype(np.int32)
    clause_sign = (clause_vars % 2).astype(np.int8)  # 1 for positive, 0 for negative
    
    # Build variable-to-clauses mapping for efficient updates
    # For each variable, store list of (clause_idx, position_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for c in range(m):
        for j in range(3):
            v = clause_var_idx[c, j]
            var_to_clauses[v].append((c, j))
    
    # State: assignment as boolean array (True = positive literal is true)
    # sat_count[c] = number of satisfied literals in clause c
    def random_assignment(rng):
        return np.array([rng.random() < 0.5 for _ in range(n)])
    
    def compute_sat_counts(assignment):
        # sat_count[c] = number of literals in clause c that are satisfied
        sat_counts = np.zeros(m, dtype=np.int32)
        for c in range(m):
            count = 0
            for j in range(3):
                v = clause_var_idx[c, j]
                sign = clause_sign[c, j]
                if sign == 1:
                    if assignment[v]:
                        count += 1
                else:
                    if not assignment[v]:
                        count += 1
            sat_counts[c] = count
        return sat_counts
    
    def is_satisfied(assignment, sat_counts):
        return np.all(sat_counts > 0)
    
    def flip_variable(var_idx, assignment, sat_counts):
        """Flip variable var_idx and update sat_counts incrementally."""
        # Determine if flipping helps or hurts each clause containing var_idx
        old_val = assignment[var_idx]
        new_val = not old_val
        
        for (c, j) in var_to_clauses[var_idx]:
            sign = clause_sign[c, j]
            # Was this literal satisfied before?
            if sign == 1:
                was_sat = old_val
                is_sat = new_val
            else:
                was_sat = not old_val
                is_sat = not new_val
            
            if was_sat and not is_sat:
                sat_counts[c] -= 1
            elif not was_sat and is_sat:
                sat_counts[c] += 1
        
        assignment[var_idx] = new_val
    
    # Main search loop with restarts
    best_assignment = None
    best_unsat = m + 1
    
    rng = random.Random(0)
    
    # Adaptive parameters
    p = 0.55  # probability of flipping a literal in the unsatisfied clause
    max_steps_per_restart = 5000
    num_restarts = 0
    
    while time.time() < deadline:
        # Random restart
        assignment = random_assignment(rng)
        sat_counts = compute_sat_counts(assignment)
        num_restarts += 1
        
        # WalkSAT loop
        for step in range(max_steps_per_restart):
            if time.time() >= deadline:
                break
            
            # Find unsatisfied clauses
            unsat_mask = sat_counts == 0
            if not np.any(unsat_mask):
                # Solved!
                return assignment.tolist()
            
            # Track best solution found so far
            unsat_count = int(np.sum(unsat_mask))
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            
            # Select a random unsatisfied clause
            unsat_indices = np.where(unsat_mask)[0]
            clause_idx = rng.choice(unsat_indices.tolist())
            
            # Choose which literal to flip
            # With probability p, flip a literal in the clause that is currently unsatisfied
            # With probability (1-p), flip a random literal in the clause
            clause_literals = list(range(3))
            rng.shuffle(clause_literals)
            
            # Score each literal by how many other clauses it would satisfy
            best_var = -1
            best_score = -1
            for j in clause_literals:
                v = clause_var_idx[clause_idx, j]
                sign = clause_sign[clause_idx, j]
                
                # Calculate score: number of clauses that would become satisfied
                # minus number of clauses that would become unsatisfied
                score = 0
                for (c2, j2) in var_to_clauses[v]:
                    if c2 == clause_idx:
                        continue
                    sign2 = clause_sign[c2, j2]
                    # Would flipping v satisfy clause c2?
                    if sign2 == 1:
                        would_satisfy = not assignment[v]
                        would_unsat = assignment[v]
                    else:
                        would_satisfy = assignment[v]
                        would_unsat = not assignment[v]
                    
                    if sat_counts[c2] == 0:
                        if would_satisfy:
                            score += 1
                    elif would_unsat:
                        score -= 1
                
                if score > best_score:
                    best_score = score
                    best_var = v
            
            # With probability p, use the best variable; otherwise random
            if rng.random() < p:
                var_to_flip = best_var
            else:
                var_to_flip = rng.choice([clause_var_idx[clause_idx, j] for j in clause_literals])
            
            flip_variable(var_to_flip, assignment, sat_counts)
        
        # Check if we solved it
        unsat_mask = sat_counts == 0
        if not np.any(unsat_mask):
            return assignment.tolist()
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    return assignment.tolist()
# EVOLVE-BLOCK-END
