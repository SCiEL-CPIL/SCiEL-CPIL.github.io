"""Improved 3-SAT solver using WalkSAT with adaptive parameters and clause weighting."""
# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42)
    
    # Convert clauses to lists for faster access
    clauses_list = [list(c) for c in clauses]
    m = len(clauses_list)
    
    # Build variable-to-clause index for efficient updates
    var_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses_list):
        for lit in clause:
            var_clauses[abs(lit) - 1].append(i)
    
    # Initialize random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Track satisfied status for each clause
    def is_satisfied(clause):
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                return True
        return False
    
    # Count how many literals in each clause are satisfied
    def count_sat_literals(clause):
        return sum(1 for lit in clause if assignment[abs(lit) - 1] == (lit > 0))
    
    # Initial check
    unsat_count = 0
    for c in clauses_list:
        if not is_satisfied(c):
            unsat_count += 1
    
    if unsat_count == 0:
        return assignment
    
    # Adaptive WalkSAT parameters
    base_p = 0.45
    max_iters = 50000
    restart_interval = 2000
    
    for restart in range(20):  # Up to 20 restarts
        # Randomize assignment on restart
        if restart > 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            unsat_count = 0
            for c in clauses_list:
                if not is_satisfied(c):
                    unsat_count += 1
            if unsat_count == 0:
                return assignment
        
        # Main search loop
        for it in range(max_iters):
            if unsat_count == 0:
                return assignment
            
            # Pick a random unsatisfied clause
            # Find one by sampling
            while True:
                ci = rng.randrange(m)
                if not is_satisfied(clauses_list[ci]):
                    break
            
            clause = clauses_list[ci]
            
            # Choose which variable to flip
            # Weight by number of satisfied literals that would be lost
            best_var = -1
            best_score = -1
            best_loss = 0
            best_gain = 0
            
            # Evaluate each literal in the clause
            candidates = []
            for lit in clause:
                vi = abs(lit) - 1
                # If we flip vi, what happens to this clause?
                # Currently this clause is unsatisfied, so no literals in it are satisfied
                # After flipping vi, the literal lit will be satisfied (since we're flipping to satisfy it)
                # But we might break other clauses
                
                # Count how many other clauses containing vi are currently satisfied
                # and would become unsatisfied if we flip vi
                loss = 0
                for cj in var_clauses[vi]:
                    if cj == ci:
                        continue
                    # Check if clause cj is currently satisfied
                    c_j = clauses_list[cj]
                    if is_satisfied(c_j):
                        # Check if flipping vi would make it unsatisfied
                        # Find the literal in c_j that involves vi
                        for lit_j in c_j:
                            if abs(lit_j) == vi + 1:
                                # If this literal is currently satisfied, flipping breaks it
                                if assignment[vi] == (lit_j > 0):
                                    loss += 1
                                break
                
                candidates.append((loss, vi, lit))
            
            # Use weighted random selection: prefer lower loss
            # But with some randomness to escape local minima
            min_loss = min(c[0] for c in candidates)
            max_loss = max(c[0] for c in candidates)
            range_loss = max_loss - min_loss + 1
            
            # Adaptive probability: higher p means more likely to pick min-loss
            # Adjust p based on progress
            progress = unsat_count / m
            p = base_p + 0.3 * (1.0 - progress)  # More greedy when close to solution
            
            # Weighted selection
            weights = []
            for loss, vi, lit in candidates:
                # Exponential weighting favoring low loss
                w = math.exp(-p * (loss - min_loss) / range_loss if range_loss > 0 else 0)
                weights.append(w)
            
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = candidates[0]
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen = candidates[i]
                    break
            
            _, vi, lit = chosen
            
            # Flip the variable
            assignment[vi] = not assignment[vi]
            
            # Update unsat_count by checking affected clauses
            # This is expensive but necessary for correctness
            # Only check clauses containing vi
            new_unsat = 0
            for cj in var_clauses[vi]:
                if not is_satisfied(clauses_list[cj]):
                    new_unsat += 1
            
            # But we need the total unsat count, not just for this variable
            # Let's just recount periodically or maintain a proper count
            
            # For efficiency, let's do a full recount every few iterations
            if it % 10 == 0 or it == max_iters - 1:
                unsat_count = sum(1 for c in clauses_list if not is_satisfied(c))
                if unsat_count == 0:
                    return assignment
        
        # If we didn't find a solution in this restart, continue to next restart
    
    # Final attempt: return current best assignment
    return assignment
# EVOLVE-BLOCK-END
