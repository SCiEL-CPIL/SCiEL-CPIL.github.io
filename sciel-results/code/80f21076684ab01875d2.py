"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Pre-compute variable to clause index mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for l in clause:
            var_to_clauses[abs(l) - 1].append(i)

    # WalkSAT parameters
    # Standard WalkSAT uses p ~ 0.5, but for planted instances, slightly higher p (randomness)
    # or lower p (greedy) can work. We'll use a standard setup.
    p = 0.5 
    max_restarts = 100
    max_steps = 200000
    timeout = 280.0 # Safety margin for 300s limit
    
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        rng = random.Random(restart * 1000003 + 1)
        
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Pre-compute clause satisfaction status and count of satisfied literals
        # sat_count[i] = number of satisfied literals in clause i
        sat_count = [0] * m
        for i, clause in enumerate(clauses):
            count = 0
            for l in clause:
                if assignment[abs(l) - 1] == (l > 0):
                    count += 1
            sat_count[i] = count
            
        # List of unsatisfied clauses
        unsat_clauses = [i for i in range(m) if sat_count[i] == 0]
        
        if not unsat_clauses:
            return assignment
            
        for step in range(max_steps):
            if time.time() - start_time > timeout:
                return assignment # Return best effort or just exit? Task says return list.
            
            if not unsat_clauses:
                return assignment
                
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat_clauses)
            clause = clauses[clause_idx]
            
            # Choose a variable to flip
            if rng.random() < p:
                # Random choice from clause
                lit = rng.choice(clause)
                var_idx = abs(lit) - 1
            else:
                # Min-conflicts: choose variable that minimizes number of newly unsatisfied clauses
                best_var = -1
                best_cost = float('inf')
                best_tie_breaker = float('inf')
                
                for lit in clause:
                    var_idx = abs(lit) - 1
                    # Calculate cost: how many currently satisfied clauses containing var_idx 
                    # would become unsatisfied if we flip var_idx?
                    # Actually, simpler: count how many clauses containing var_idx 
                    # have exactly 1 satisfied literal (which is var_idx's current value? No, 
                    # we are flipping, so we check if the current assignment satisfies them 
                    # ONLY via this variable).
                    
                    # Wait, standard WalkSAT min-conflict step:
                    # For each var in clause, count the number of *other* clauses 
                    # (not the current one) that contain this var and are currently 
                    # satisfied ONLY by this var's current value.
                    # If we flip, those clauses become unsatisfied.
                    # We want to minimize this number.
                    
                    cost = 0
                    for c_idx in var_to_clauses[var_idx]:
                        if c_idx == clause_idx:
                            continue
                        # Check if clause c_idx is satisfied only by var_idx
                        # If sat_count[c_idx] == 1, and the only satisfied literal is var_idx,
                        # then flipping var_idx will make it unsatisfied.
                        if sat_count[c_idx] == 1:
                            # Check if var_idx is the satisfied one
                            c = clauses[c_idx]
                            for l in c:
                                v = abs(l) - 1
                                if assignment[v] == (l > 0):
                                    if v == var_idx:
                                        cost += 1
                                    break
                                    
                    # Tie-breaking: random
                    tie_breaker = rng.random()
                    
                    if cost < best_cost or (cost == best_cost and tie_breaker < best_tie_breaker):
                        best_cost = cost
                        best_tie_breaker = tie_breaker
                        best_var = var_idx
                        
                var_idx = best_var
                
            # Flip the variable
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Update sat_count and unsat_clauses
            # For each clause containing var_idx:
            for c_idx in var_to_clauses[var_idx]:
                c = clauses[c_idx]
                # Determine if the literal of var_idx in this clause was satisfied before
                lit_val = None
                for l in c:
                    if abs(l) - 1 == var_idx:
                        lit_val = l
                        break
                
                was_sat = (old_val == (lit_val > 0))
                is_sat = (not old_val == (lit_val > 0)) # new value is not old_val
                
                if was_sat and not is_sat:
                    # Clause became unsatisfied
                    sat_count[c_idx] -= 1
                    if sat_count[c_idx] == 0:
                        # Add to unsat list if not already there? 
                        # It was satisfied before, so it wasn't in unsat_clauses.
                        unsat_clauses.append(c_idx)
                elif not was_sat and is_sat:
                    # Clause became satisfied
                    sat_count[c_idx] += 1
                    if sat_count[c_idx] == 1:
                        # It was unsatisfied (count 0), now count 1.
                        # Remove from unsat_clauses
                        try:
                            unsat_clauses.remove(c_idx)
                        except ValueError:
                            pass
                # If both was_sat and is_sat (impossible for single var flip) or both not, count doesn't change.
                # Wait, if var_idx was not the satisfied literal, sat_count doesn't change.
                # If var_idx was satisfied, and we flip, it becomes unsatisfied.
                # If var_idx was not satisfied, and we flip, it becomes satisfied.
                
    return assignment
# EVOLVE-BLOCK-END
