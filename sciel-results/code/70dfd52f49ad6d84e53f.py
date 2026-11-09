"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    # Planted 3-SAT solver with WalkSAT + restarts and time guard.
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    deadline = time.time() + 290  # safety margin before 300s kill
    rng = random.Random(12345)
    
    # Preprocess: convert clauses to (var_indices, signs) for faster access
    # Each clause: list of signed literals (1-based)
    # We'll keep as-is but precompute abs and sign
    # For speed, let's use numpy-like approach but with lists
    
    # WalkSAT with restarts
    # Strategy: 
    #   - Random restarts with increasing step budget
    #   - Weighted literal selection: prefer literals that appear in more unsat clauses
    #   - Track best assignment (fewest unsat clauses)
    
    # Precompute literal occurrence counts for weighting
    lit_count = {}
    for c in clauses:
        for l in c:
            lit_count[l] = lit_count.get(l, 0) + 1
    
    # Also count by variable
    var_count = {}
    for c in clauses:
        for l in c:
            v = abs(l)
            var_count[v] = var_count.get(v, 0) + 1
    
    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        cnt = 0
        for c in clauses:
            sat = False
            for l in c:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses."""
        for c in clauses:
            sat = False
            for l in c:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    # Try multiple random restarts
    best_assignment = None
    best_unsat = m + 1
    restart_count = 0
    
    # First, try a simple random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    unsat = count_unsat(assignment)
    if unsat == 0:
        return assignment
    best_assignment = assignment[:]
    best_unsat = unsat
    
    # WalkSAT with restarts
    # Use a p-parameter for choosing between "satisfying" and "random" literal flips
    # p = 0.5 means 50% chance to pick a literal that satisfies the clause
    # Adjust p based on progress
    
    max_restarts = 200
    base_steps = 500
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Restart: new random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat = count_unsat(assignment)
        if unsat == 0:
            return assignment
        
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
        
        # Walk for a certain number of steps
        # Increase steps with restart count to allow deeper search
        steps = base_steps + restart * 100
        if steps > 5000:
            steps = 5000
        
        p = 0.55  # probability to pick a literal that satisfies the current clause
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for c in clauses:
                sat = False
                for l in c:
                    v = abs(l) - 1
                    if assignment[v] == (l > 0):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(c)
            
            if not unsat_clauses:
                return assignment
            
            # Pick a random unsatisfied clause
            clause = unsat_clauses[rng.randrange(len(unsat_clauses))]
            
            # Count how many literals in this clause are currently satisfied
            # (i.e., flipping them would unsatisfy the clause)
            # We want to flip a literal. With probability p, pick one that
            # makes the clause satisfied (i.e., currently false in the clause)
            
            # For each literal in the clause, determine if flipping it satisfies the clause
            # A literal l is currently true if assignment[abs(l)-1] == (l > 0)
            # Flipping it makes it false. So to satisfy the clause by flipping,
            # we need to flip a literal that is currently false.
            
            false_lits = []
            true_lits = []
            for l in clause:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    true_lits.append(l)
                else:
                    false_lits.append(l)
            
            if false_lits:
                # With probability p, pick from false_lits (satisfying flip)
                # Otherwise, pick from all literals
                if rng.random() < p:
                    lit = false_lits[rng.randrange(len(false_lits))]
                else:
                    lit = clause[rng.randrange(len(clause))]
            else:
                # All literals are true, so any flip will unsatisfy this clause
                # Pick randomly
                lit = clause[rng.randrange(len(clause))]
            
            # Flip the literal
            v = abs(lit) - 1
            assignment[v] = not assignment[v]
            
            # Check if we solved it (only every few steps for speed)
            if step % 10 == 0:
                unsat = count_unsat(assignment)
                if unsat == 0:
                    return assignment
                if unsat < best_unsat:
                    best_unsat = unsat
                    best_assignment = assignment[:]
        
        # After walk, check final state
        unsat = count_unsat(assignment)
        if unsat == 0:
            return assignment
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
    
    # Return best assignment found
    if best_assignment is not None:
        # Final check
        if is_satisfied(best_assignment):
            return best_assignment
        return best_assignment
    else:
        return [False] * n

# EVOLVE-BLOCK-END
