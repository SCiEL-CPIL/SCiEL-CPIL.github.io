"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s kill
    
    # Precompute clause literals as (var_idx, polarity) for faster access
    # var_idx is 0-based, polarity is True for positive, False for negative
    clauses_opt = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 12345 + 67890)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize unsat count
        unsat_count = 0
        for c in clauses_opt:
            sat = False
            for var_idx, pol in c:
                if assignment[var_idx] == pol:
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT with adaptive noise
        steps_per_restart = 15000
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for c in clauses_opt:
                sat = False
                for var_idx, pol in c:
                    if assignment[var_idx] == pol:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(c)
            
            if not unsat_clauses:
                return assignment
            
            if len(unsat_clauses) < best_unsat_count:
                best_unsat_count = len(unsat_clauses)
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            clause = unsat_clauses[rng.randrange(len(unsat_clauses))]
            
            # Count how many literals in this clause would be satisfied if flipped
            # and how many other clauses would be broken
            best_var = None
            best_score = float('inf')
            
            for var_idx, pol in clause:
                # If we flip this variable, what happens?
                # Current value: assignment[var_idx]
                # New value: not assignment[var_idx]
                # This literal is satisfied if assignment[var_idx] == pol
                # After flip, this literal is satisfied if (not assignment[var_idx]) == pol
                
                # Count satisfied literals in this clause after flip
                sat_in_clause = 0
                for v, p in clause:
                    if v == var_idx:
                        if (not assignment[v]) == p:
                            sat_in_clause += 1
                    else:
                        if assignment[v] == p:
                            sat_in_clause += 1
                
                # Score: number of unsatisfied literals in this clause after flip (lower is better)
                score = len(clause) - sat_in_clause
                
                if score < best_score:
                    best_score = score
                    best_var = var_idx
            
            if best_var is None:
                continue
            
            # With some probability, flip a random variable from the clause (noise)
            if rng.random() < 0.25:
                var_idx, pol = clause[rng.randrange(len(clause))]
            else:
                var_idx = best_var
            
            assignment[var_idx] = not assignment[var_idx]
        
        if time.time() > deadline:
            break
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: return current assignment
    return assignment
# EVOLVE-BLOCK-END
