"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline guard
    deadline = time.time() + 290.0  # Leave 10s margin for 300s limit
    
    # Preprocess: convert clauses to 0-based indices and store signs
    # For speed, we'll use numpy arrays
    # clause_vars[i] = array of variable indices (0-based)
    # clause_signs[i] = array of booleans (True if positive literal)
    
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_ = []
        signs = []
        for lit in c:
            v = abs(lit) - 1
            s = lit > 0
            vars_.append(v)
            signs.append(s)
        clause_vars.append(vars_)
        clause_signs.append(signs)
    
    # Variable occurrence counts for weighting
    var_pos_count = [0] * n
    var_neg_count = [0] * n
    for c in clauses:
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                var_pos_count[v] += 1
            else:
                var_neg_count[v] += 0  # Not needed for basic WalkSAT, but good for stats
    var_total = [var_pos_count[i] for i in range(n)]
    
    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses."""
        for i in range(num_clauses):
            vars_ = clause_vars[i]
            signs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[vars_[j]] == signs[j]:
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for i in range(num_clauses):
            vars_ = clause_vars[i]
            signs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[vars_[j]] == signs[j]:
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def unsat_clauses(assignment):
        """Return list of indices of unsatisfied clauses."""
        unsat = []
        for i in range(num_clauses):
            vars_ = clause_vars[i]
            signs = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[vars_[j]] == signs[j]:
                    sat = True
                    break
            if not sat:
                unsat.append(i)
        return unsat
    
    best_assignment = None
    best_unsat = float('inf')
    
    rng = random.Random()
    
    # Try multiple restarts
    max_restarts = 500
    steps_per_restart = 10000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 1.0:
            break
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # WalkSAT with adaptive noise
        for step in range(steps_per_restart):
            if time.time() > deadline - 1.0:
                break
            
            # Find unsatisfied clauses
            unsat = unsat_clauses(assignment)
            if not unsat:
                # Found satisfying assignment
                return assignment
            
            # Track best
            if len(unsat) < best_unsat:
                best_unsat = len(unsat)
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat)
            vars_ = clause_vars[clause_idx]
            signs = clause_signs[clause_idx]
            
            # Count how many literals in this clause are false
            # and how many true
            false_vars = []
            for j in range(3):
                if assignment[vars_[j]] != signs[j]:
                    false_vars.append(vars_[j])
            
            # Pick a variable to flip from the false literals
            if false_vars:
                v = rng.choice(false_vars)
            else:
                # This shouldn't happen if clause is unsatisfied, but just in case
                v = rng.choice(vars_)
            
            # Flip the variable
            assignment[v] = not assignment[v]
        
        # If we didn't find a solution in this restart, continue
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: random assignment
        return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
