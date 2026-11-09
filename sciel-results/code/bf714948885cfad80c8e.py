"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Precompute clause literals as (var_index, polarity) for speed
    # var_index is 0-based, polarity is +1 or -1
    cl = []
    for c in clauses:
        cl.append([(abs(l) - 1, 1 if l > 0 else -1) for l in c])
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s/330s limit
    
    # Best known assignment (by number of satisfied clauses)
    best_sat_count = -1
    best_assign = None
    
    # WalkSAT with restarts and adaptive noise
    # For planted 3-SAT with 50 vars, the formula is likely at or near threshold
    # Use a combination of WalkSAT and local search with restarts
    
    rng = random.Random(0)
    
    # Try multiple restarts with increasing step budgets
    # For planted instances, they are usually easy if below threshold
    # but can be hard near threshold
    
    # Strategy: 
    # 1. Try WalkSAT with standard parameters
    # 2. If not solved, try with different noise levels
    # 3. Track best assignment found
    
    # Number of variables is 50, so we can afford many restarts
    max_restarts = 500
    base_steps = 5000
    noise_schedule = [0.25, 0.5, 0.75, 0.35, 0.65, 0.45, 0.55, 0.3, 0.7, 0.4]
    
    def is_satisfied(assign, clause):
        for var, pol in clause:
            if assign[var] == (pol > 0):
                return True
        return False
    
    def count_satisfied(assign):
        return sum(1 for c in cl if is_satisfied(assign, c))
    
    def walksat(assign, noise, max_steps, rng):
        """Run WalkSAT, return (satisfied, assign, sat_count)"""
        for step in range(max_steps):
            # Check if solved
            unsat_clauses = [i for i in range(m) if not is_satisfied(assign, cl[i])]
            if not unsat_clauses:
                return True, assign, m
            
            # Pick a random unsatisfied clause
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            clause = cl[ci]
            
            # Count for each literal how many clauses it would satisfy
            best_var = None
            best_gain = -1
            best_tie_break = 0
            
            for var, pol in clause:
                # If we flip var to satisfy this literal, how many other unsat clauses
                # would we satisfy?
                gain = 0
                for j in range(m):
                    cj = cl[j]
                    if is_satisfied(assign, cj):
                        continue
                    for v2, p2 in cj:
                        if v2 == var:
                            # If flipping var to satisfy our literal also satisfies cj
                            if p2 == pol:
                                gain += 1
                            break
                
                if gain > best_gain or (gain == best_gain and rng.random() < 0.5):
                    best_gain = gain
                    best_var = var
                    best_tie_break = rng.random()
            
            # With probability noise, flip a random variable in the clause
            if rng.random() < noise:
                var, pol = clause[rng.randrange(len(clause))]
            else:
                var = best_var
            
            # Flip the variable
            # We want to set assign[var] such that the literal (var, pol) is satisfied
            # If pol > 0, we want assign[var] = True
            # If pol < 0, we want assign[var] = False
            # But in WalkSAT, we typically just flip
            # Let's flip to the value that satisfies the literal
            target_val = (pol > 0)
            assign[var] = target_val
        
        sat_count = count_satisfied(assign)
        return False, assign, sat_count
    
    # Main loop with restarts
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Choose noise level
        noise = noise_schedule[restart % len(noise_schedule)]
        
        # Initialize with random assignment
        assign = [rng.random() < 0.5 for _ in range(n)]
        
        # Run WalkSAT
        solved, assign, sat_count = walksat(assign, noise, base_steps, rng)
        
        if sat_count > best_sat_count:
            best_sat_count = sat_count
            best_assign = list(assign)
        
        if solved:
            return assign
    
    # If we didn't solve, return the best assignment found
    if best_assign is not None:
        return best_assign
    
    # Fallback: random assignment
    return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
