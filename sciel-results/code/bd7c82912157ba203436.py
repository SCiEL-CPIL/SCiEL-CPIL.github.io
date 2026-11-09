"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    # Improved WalkSAT / MiniSAT-like local search with restarts and deadline guard.
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 300s wall-clock limit, use 295s for safety margin.
    deadline = time.time() + 295.0
    
    # Pre-compute clause sizes for efficiency
    clause_sizes = [len(c) for c in clauses]
    
    # Convert clauses to list of (var, polarity) for faster access
    # var is 0-based, polarity is 1 for positive, 0 for negative
    clauses_v = [[(abs(l)-1, 1 if l > 0 else 0) for l in c] for c in clauses]
    
    # Build index: for each variable, which clauses contain it
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_v):
        for var, pol in c:
            var_to_clauses[var].append(ci)
    
    def is_satisfied(assignment, clause):
        for var, pol in clause:
            if assignment[var] == (pol == 1):
                return True
        return False
    
    def all_satisfied(assignment):
        for c in clauses_v:
            if not is_satisfied(assignment, c):
                return False
        return True
    
    def count_unsat(assignment):
        cnt = 0
        for c in clauses_v:
            if not is_satisfied(assignment, c):
                cnt += 1
        return cnt
    
    # Track best assignment found so far
    best_assignment = None
    best_unsat_count = float('inf')
    
    rng = random.Random(0)
    
    # Multiple restarts with increasing step budgets
    max_restarts = 200
    base_steps = 5000
    max_steps = 50000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize assignment randomly
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial unsat count
        unsat_count = count_unsat(assignment)
        
        # Update best if better
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            if unsat_count == 0:
                return assignment
        
        # Determine step budget for this restart (increasing with restarts)
        steps = min(base_steps + restart * 500, max_steps)
        
        # WalkSAT loop
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break
            
            # Find an unsatisfied clause
            unsat_clauses = []
            for ci, c in enumerate(clauses_v):
                if not is_satisfied(assignment, c):
                    unsat_clauses.append(ci)
            
            if not unsat_clauses:
                return assignment
            
            # Update best if this assignment is better
            current_unsat = len(unsat_clauses)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            clause = clauses_v[ci]
            
            # Score each literal by how many clauses would be satisfied if flipped
            best_var = -1
            best_score = -1
            best_flip = False
            
            for var, pol in clause:
                # If we flip var, what happens?
                # Current value: assignment[var]
                # New value: not assignment[var]
                new_val = not assignment[var]
                
                # Count clauses that would become satisfied
                # and clauses that would become unsatisfied
                score = 0
                for cj in var_to_clauses[var]:
                    c_j = clauses_v[cj]
                    # Check if clause cj is currently satisfied
                    cur_sat = False
                    for v, p in c_j:
                        if assignment[v] == (p == 1):
                            cur_sat = True
                            break
                    
                    # Check if clause cj would be satisfied after flip
                    new_sat = False
                    for v, p in c_j:
                        if v == var:
                            if new_val == (p == 1):
                                new_sat = True
                                break
                        else:
                            if assignment[v] == (p == 1):
                                new_sat = True
                                break
                    
                    if cur_sat and not new_sat:
                        score -= 1
                    elif not cur_sat and new_sat:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_var = var
                    best_flip = new_val
            
            # With some probability, pick a random literal from the clause (for exploration)
            if rng.random() < 0.3:
                var, pol = rng.choice(clause)
                assignment[var] = not assignment[var]
            else:
                if best_var >= 0:
                    assignment[best_var] = best_flip
        
        # After steps exhausted, check if solved
        if all_satisfied(assignment):
            return assignment
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: return last assignment
    return assignment
# EVOLVE-BLOCK-END
