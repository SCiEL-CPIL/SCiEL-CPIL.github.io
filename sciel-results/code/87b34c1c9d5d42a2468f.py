"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    """
    Solve a planted 3-SAT instance using an optimized WalkSAT algorithm:
    1. Phase saving heuristic initialization
    2. Pre-computed variable-clause incidence lists
    3. Incremental tracking of unsatisfied clauses
    4. Efficient flip selection based on clause gain/loss
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Build literal occurrence counts for phase saving
    pos_count = [0] * n
    neg_count = [0] * n
    for clause in clauses:
        for lit in clause:
            if lit > 0:
                pos_count[lit - 1] += 1
            else:
                neg_count[-lit - 1] += 1
    
    # Initial assignment using phase saving heuristic
    assignment = [pos_count[i] > neg_count[i] for i in range(n)]
    
    # Pre-compute clause data: list of (var_idx, sign) for each clause
    # sign is True if positive literal, False if negative
    clause_data = []
    for clause in clauses:
        cdata = []
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = lit > 0
            cdata.append((var_idx, sign))
        clause_data.append(cdata)
    
    # Pre-compute variable to clause incidence
    # var_clauses[v] = list of (clause_idx, sign_in_clause)
    var_clauses = [[] for _ in range(n)]
    for ci, cdata in enumerate(clause_data):
        for var_idx, sign in cdata:
            var_clauses[var_idx].append((ci, sign))
    
    m = len(clauses)
    
    # Helper to check if a clause is satisfied
    def is_satisfied(ci):
        cdata = clause_data[ci]
        for var_idx, sign in cdata:
            if assignment[var_idx] == sign:
                return True
        return False
    
    # Initialize unsatisfied clause list
    unsat_clauses = [i for i in range(m) if not is_satisfied(i)]
    
    if not unsat_clauses:
        return assignment
    
    # WalkSAT parameters
    total_budget = 1000000  # Total flip operations
    num_restarts = 20
    flips_per_restart = total_budget // num_restarts
    
    for restart in range(num_restarts):
        # Initialize assignment
        if restart == 0:
            # Keep phase-saving assignment
            pass
        else:
            rng_init = random.Random(42 + restart)
            # Random initialization with some phase saving bias
            ps = [pos_count[i] > neg_count[i] for i in range(n)]
            for i in range(n):
                if rng_init.random() < 0.2:
                    assignment[i] = ps[i]
                else:
                    assignment[i] = rng_init.random() < 0.5
            
            # Rebuild unsat list
            unsat_clauses = [i for i in range(m) if not is_satisfied(i)]
            
            if not unsat_clauses:
                return assignment
        
        rng = random.Random(100 + restart)
        
        # Local search loop
        for step in range(flips_per_restart):
            if not unsat_clauses:
                return assignment
            
            # Select a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cdata = clause_data[ci]
            
            # Calculate flip scores for variables in this clause
            # Score = - (num other unsat clauses fixed) + (num sat clauses broken)
            # We want to minimize score.
            
            best_var_idx = -1
            best_score = float('inf')
            
            for var_idx, sign in cdata:
                # Current value is not sign. New value will be sign.
                # Check impact on other clauses containing var_idx.
                
                gain = 0
                loss = 0
                
                for other_ci, other_sign in var_clauses[var_idx]:
                    if other_ci == ci:
                        continue
                    
                    # Check if other_ci is currently satisfied
                    other_sat = False
                    for v, s in clause_data[other_ci]:
                        if assignment[v] == s:
                            other_sat = True
                            break
                    
                    if other_sat:
                        # Check if var_idx was the ONLY satisfier
                        # Current value of var_idx is not sign.
                        # If (not sign) == other_sign, then var_idx currently satisfies other_ci.
                        if (not sign) == other_sign:
                            # Check if any other literal satisfies other_ci
                            has_other_sat = False
                            for v, s in clause_data[other_ci]:
                                if v == var_idx:
                                    continue
                                if assignment[v] == s:
                                    has_other_sat = True
                                    break
                            if not has_other_sat:
                                loss += 1
                    else:
                        # other_ci is unsatisfied
                        # If new value satisfies the literal, it's a gain
                        if sign == other_sign:
                            gain += 1
                
                score = loss - gain
                if score < best_score:
                    best_score = score
                    best_var_idx = var_idx
            
            # Noise: with some probability, flip a random variable from the clause
            noise_prob = 0.2
            if rng.random() < noise_prob:
                var_idx, sign = rng.choice(cdata)
            else:
                var_idx = best_var_idx
            
            # Perform flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update unsat_clauses list
            # Identify affected clauses (those containing var_idx)
            affected_set = set(ci_aff for ci_aff, _ in var_clauses[var_idx])
            
            new_unsat = []
            # Keep unaffected unsat clauses
            for ci_u in unsat_clauses:
                if ci_u not in affected_set:
                    new_unsat.append(ci_u)
            
            # Add affected clauses that are unsatisfied
            for ci_a in affected_set:
                if not is_satisfied(ci_a):
                    new_unsat.append(ci_a)
            
            unsat_clauses = new_unsat
            
            if not unsat_clauses:
                return assignment
        
        # End of restart loop
        if not unsat_clauses:
            return assignment
    
    # Final check
    if not unsat_clauses:
        return assignment
    
    return assignment
# EVOLVE-BLOCK-END
