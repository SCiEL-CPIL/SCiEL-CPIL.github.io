"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _walksat_step(assignment, clause, rng, weight, n):
    """Pick a literal from clause to flip, weighted by weight and random noise."""
    # For each literal in clause, compute a score
    # We want to flip a literal that is currently false (to make clause true)
    # Weight of variable: how many clauses it appears in (more constraints = more important)
    best_literal = None
    best_score = -1
    for l in clause:
        var = abs(l) - 1
        current_val = assignment[var]
        literal_is_true = (l > 0 and current_val) or (l < 0 and not current_val)
        if not literal_is_true:
            # This literal is false; flipping it will make it true
            # Score: weight of variable + random noise
            score = weight[var] + rng.random() * 2.0
            if score > best_score:
                best_score = score
                best_literal = l
        else:
            # Literal is already true; flipping it might break other clauses
            # Lower probability
            score = rng.random() * 0.1
            if score > best_score:
                best_score = score
                best_literal = l
    if best_literal is None:
        best_literal = rng.choice(clause)
    return best_literal


def _walksat(instance, rng, max_steps, deadline):
    """WalkSAT with dynamic weight update."""
    n, clauses = instance["n"], instance["clauses"]
    
    # Initialize assignment randomly
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Build weight array: frequency of each variable
    weight = np.zeros(n)
    for clause in clauses:
        for l in clause:
            weight[abs(l) - 1] += 1
    
    # Count unsatisfied clauses
    def count_unsat():
        count = 0
        for c in clauses:
            sat = False
            for l in c:
                if (l > 0 and assignment[abs(l) - 1]) or (l < 0 and not assignment[abs(l) - 1]):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    best_unsat = count_unsat()
    best_assignment = assignment[:]
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Find an unsatisfied clause
        unsat_clause = None
        for c in clauses:
            sat = False
            for l in c:
                if (l > 0 and assignment[abs(l) - 1]) or (l < 0 and not assignment[abs(l) - 1]):
                    sat = True
                    break
            if not sat:
                unsat_clause = c
                break
        
        if unsat_clause is None:
            # Satisfied!
            return assignment
        
        # Pick a literal to flip
        literal = _walksat_step(assignment, unsat_clause, rng, weight, n)
        var = abs(literal) - 1
        assignment[var] = not assignment[var]
        
        # Update weights: increase weight of variables in unsat clause
        for l in unsat_clause:
            weight[abs(l) - 1] += 0.1
        # Decay weights slightly to prevent explosion
        weight *= 0.999
        
        # Track best
        current_unsat = count_unsat()
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]
    
    return best_assignment


def solve(instance):
    """
    Solve a planted 3-SAT instance using WalkSAT with multiple restarts.
    Returns a list of booleans (assignment) of length n.
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline: 330s total, leave 10s margin
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Use a fixed seed for reproducibility, but allow variation
    rng = random.Random(42)
    
    # Multiple restarts strategy
    # Each restart uses a fraction of the remaining time
    max_steps_per_restart = 5000
    min_time_per_restart = 0.5
    
    best_result = None
    best_unsat = float('inf')
    
    # First, try a quick random assignment check
    for attempt in range(20):
        if time.time() > deadline:
            break
        test_assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_count = 0
        for c in clauses:
            sat = False
            for l in c:
                if (l > 0 and test_assignment[abs(l) - 1]) or (l < 0 and not test_assignment[abs(l) - 1]):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        if unsat_count == 0:
            return test_assignment
    
    # Main search loop with restarts
    restart_count = 0
    while time.time() < deadline:
        # Determine steps for this restart based on remaining time
        remaining_time = deadline - time.time()
        if remaining_time < min_time_per_restart:
            break
        
        # Use a portion of remaining time for this restart
        steps = max(1000, min(max_steps_per_restart, int(remaining_time * 2000)))
        
        result = _walksat(instance, rng, steps, deadline)
        
        if result is None:
            break
        
        # Count unsatisfied clauses in result
        unsat_count = 0
        for c in clauses:
            sat = False
            for l in c:
                if (l > 0 and result[abs(l) - 1]) or (l < 0 and not result[abs(l) - 1]):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        
        if unsat_count == 0:
            return result
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_result = result[:]
        
        restart_count += 1
        
        # If we've done many restarts, increase step budget
        if restart_count % 5 == 0:
            max_steps_per_restart = int(max_steps_per_restart * 1.2)
    
    # Return best result found
    if best_result is not None:
        return best_result
    
    # Fallback: random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
