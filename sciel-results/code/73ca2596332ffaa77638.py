# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Preprocess: map literals to indices
    # literal > 0 means variable literal is True, literal < 0 means variable -literal is False (i.e., variable literal is False)
    # We'll use 0-based indexing internally.
    
    # Build clause list as list of (var_idx, polarity) where polarity is True for positive, False for negative
    # Actually, let's keep it simple: for each clause, store list of (var, is_positive)
    # var is 0-based, is_positive is True if literal > 0
    
    processed_clauses = []
    for clause in clauses:
        proc = []
        for lit in clause:
            var = abs(lit) - 1
            is_pos = lit > 0
            proc.append((var, is_pos))
        processed_clauses.append(proc)
    
    # Build variable-to-clauses mapping
    var_to_clauses = defaultdict(list)
    for ci, clause in enumerate(processed_clauses):
        for var, is_pos in clause:
            var_to_clauses[var].append((ci, is_pos))
    
    # Initialize assignment randomly
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    assignment = np.random.rand(n) < 0.5
    
    # Precompute clause satisfaction status
    # For each clause, track how many literals are satisfied
    clause_sat_count = np.zeros(m, dtype=np.int32)
    
    def update_clause_sat(ci, var, is_pos, old_val, new_val):
        """Update clause satisfaction count when variable changes."""
        # If the literal in this clause for this var was satisfied before, decrement
        # If it will be satisfied after, increment
        # A literal (var, is_pos) is satisfied if assignment[var] == is_pos
        was_sat = (old_val == is_pos)
        is_sat_now = (new_val == is_pos)
        if was_sat and not is_sat_now:
            clause_sat_count[ci] -= 1
        elif not was_sat and is_sat_now:
            clause_sat_count[ci] += 1
    
    # Initialize clause_sat_count
    for ci, clause in enumerate(processed_clauses):
        for var, is_pos in clause:
            if assignment[var] == is_pos:
                clause_sat_count[ci] += 1
    
    # Find unsatisfied clauses
    def get_unsat_clauses():
        return np.where(clause_sat_count == 0)[0]
    
    # WalkSAT with improvements
    # Use a phase: random restarts with increasing flip budgets
    # Also use a weighted random choice for variable selection
    
    max_restarts = 50
    base_flips = 1000
    flip_growth = 1.5
    
    for restart in range(max_restarts):
        # Reset assignment randomly
        assignment = np.random.rand(n) < 0.5
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for ci, clause in enumerate(processed_clauses):
            for var, is_pos in clause:
                if assignment[var] == is_pos:
                    clause_sat_count[ci] += 1
        
        flips_budget = int(base_flips * (flip_growth ** restart))
        
        for flip_iter in range(flips_budget):
            unsat_idx = get_unsat_clauses()
            if len(unsat_idx) == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            ci = unsat_idx[np.random.randint(len(unsat_idx))]
            clause = processed_clauses[ci]
            
            # Compute score for each literal in the clause
            # Score = number of other clauses that would be satisfied if we flip this variable
            # This is the classic WalkSAT heuristic
            
            best_var = -1
            best_score = -1
            best_is_pos = False
            
            for var, is_pos in clause:
                # If we flip var, what happens?
                # The literal (var, is_pos) in this clause becomes satisfied (since clause is currently unsat, none of its literals are satisfied, so flipping var to is_pos will satisfy this literal)
                # Wait, that's not right. The clause is unsatisfied, meaning no literal in it is satisfied.
                # If we set assignment[var] = is_pos, then this literal becomes satisfied.
                # But we need to count how many OTHER clauses become satisfied.
                
                # For each other clause that contains var, check if flipping var would satisfy it
                score = 0
                for other_ci, other_is_pos in var_to_clauses[var]:
                    if other_ci == ci:
                        continue
                    # Check if this other clause is currently unsatisfied
                    if clause_sat_count[other_ci] == 0:
                        # If we set assignment[var] = is_pos, does this satisfy the literal in other clause?
                        # The literal in other clause is (var, other_is_pos)
                        # It becomes satisfied if is_pos == other_is_pos
                        if is_pos == other_is_pos:
                            score += 1
                
                # Add a small random perturbation
                score += np.random.rand() * 0.1
                
                if score > best_score:
                    best_score = score
                    best_var = var
                    best_is_pos = is_pos
            
            if best_var == -1:
                # Should not happen, but fallback
                var, is_pos = clause[0]
                best_var = var
                best_is_pos = is_pos
            
            # Flip the variable
            old_val = assignment[best_var]
            new_val = best_is_pos
            
            if old_val != new_val:
                # Update clause_sat_count for all clauses containing best_var
                for other_ci, other_is_pos in var_to_clauses[best_var]:
                    was_sat = (old_val == other_is_pos)
                    is_sat_now = (new_val == other_is_pos)
                    if was_sat and not is_sat_now:
                        clause_sat_count[other_ci] -= 1
                    elif not was_sat and is_sat_now:
                        clause_sat_count[other_ci] += 1
                assignment[best_var] = new_val
    
    # Return the last assignment
    return assignment.tolist()

# EVOLVE-BLOCK-END
