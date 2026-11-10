"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    """
    WalkSAT with adaptive clause weighting, restarts, and a time deadline.
    Returns a list of booleans of length n.
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline: 330s wall-clock limit, use 320s as safety margin
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Pre-process clauses for faster evaluation
    # Convert clauses to (pos_vars, neg_vars) tuples for speed
    # pos_vars: list of variables that must be True
    # neg_vars: list of variables that must be False
    processed_clauses = []
    for clause in clauses:
        pos = []
        neg = []
        for l in clause:
            if l > 0:
                pos.append(l - 1)
            else:
                neg.append(-l - 1)
        processed_clauses.append((pos, neg))
    
    num_clauses = len(processed_clauses)
    
    def is_satisfied(assignment, clause_idx):
        pos, neg = processed_clauses[clause_idx]
        for v in pos:
            if assignment[v]:
                return True
        for v in neg:
            if not assignment[v]:
                return True
        return False
    
    def count_unsat(assignment):
        count = 0
        for i in range(num_clauses):
            if not is_satisfied(assignment, i):
                count += 1
        return count
    
    def get_unsat_clauses(assignment):
        unsat = []
        for i in range(num_clauses):
            if not is_satisfied(assignment, i):
                unsat.append(i)
        return unsat
    
    # Adaptive WalkSAT parameters
    p = 0.5  # probability of flipping a variable in the clause (vs random)
    max_steps_per_restart = 5000
    
    best_assignment = None
    best_unsat_count = num_clauses + 1
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 7919 + 13)
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        current_unsat = count_unsat(assignment)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        if current_unsat == 0:
            return assignment
        
        # Walk phase
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            if current_unsat == 0:
                return assignment
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            unsat_clause = None
            # Find an unsatisfied clause
            for _ in range(10):
                ci = rng.randrange(num_clauses)
                if not is_satisfied(assignment, ci):
                    unsat_clause = ci
                    break
            if unsat_clause is None:
                # Fallback: scan
                for ci in range(num_clauses):
                    if not is_satisfied(assignment, ci):
                        unsat_clause = ci
                        break
            if unsat_clause is None:
                break
            
            pos, neg = processed_clauses[unsat_clause]
            vars_in_clause = pos + neg
            
            if rng.random() < p:
                # Flip a variable in the clause
                v = vars_in_clause[rng.randrange(len(vars_in_clause))]
                assignment[v] = not assignment[v]
            else:
                # Flip a random variable
                v = rng.randrange(n)
                assignment[v] = not assignment[v]
            
            # Recompute unsat count efficiently: only clauses containing v can change
            # But for simplicity, recompute all (n=50, small)
            new_unsat = count_unsat(assignment)
            
            # If we made things worse and it's a heuristic flip, consider undoing
            if new_unsat > current_unsat + 2:
                # Undo the flip
                assignment[v] = not assignment[v]
                new_unsat = current_unsat
            else:
                current_unsat = new_unsat
        
        if time.time() > deadline:
            break
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
