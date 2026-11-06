"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: 330s wall-clock, use 325s for safety
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Preprocess clauses to 0-based with sign
    # clause_vars[i] = list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = []
        for l in c:
            var = abs(l) - 1
            pos = l > 0
            cv.append((var, pos))
        clause_vars.append(cv)
    
    # Variable occurrence lists: var_occurs[var] = list of (clause_idx, is_positive_in_clause)
    var_occurs = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var, pos in cv:
            var_occurs[var].append((ci, pos))
    
    best_assignment = None
    best_unsat_count = m + 1
    
    # Multiple restarts with increasing flip budgets
    restart_count = 0
    base_budget = 50000
    max_budget = 500000
    
    while time.time() < deadline - 5.0:
        restart_count += 1
        rng = random.Random(restart_count * 12345)
        
        # Initialize assignment: random, but bias slightly
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Track unsat count for each clause (0 or 1)
        clause_unsat = [0] * m
        # For each clause, track how many literals are false (0 = all true/satisfied, 1 = all false/unsat)
        # Actually for 3-SAT, a clause is unsat only if ALL literals are false
        # Let's track: for each clause, number of false literals
        clause_false_count = [0] * m
        
        for ci, cv in enumerate(clause_vars):
            fc = 0
            for var, pos in cv:
                if assignment[var] != pos:
                    fc += 1
            clause_false_count[ci] = fc
            if fc == len(cv):
                clause_unsat[ci] = 1
        
        total_unsat = sum(clause_unsat)
        if total_unsat == 0:
            return assignment
        
        if total_unsat < best_unsat_count:
            best_unsat_count = total_unsat
            best_assignment = assignment[:]
        
        # Flip budget for this restart
        budget = min(base_budget * (1 + restart_count * 0.5), max_budget)
        
        # Adaptive noise: start high, decrease
        initial_noise = 0.5
        noise_decay = 0.9995
        
        for step in range(budget):
            if time.time() > deadline - 10.0:
                break
            
            if total_unsat == 0:
                return assignment
            
            # Find an unsatisfied clause
            # Use a random unsat clause
            # Build list of unsat clauses occasionally for efficiency
            # For speed, just pick a random clause and check
            # Better: maintain a list of unsat clause indices
            
            # Simple approach: pick random clause, if not unsat, try a few more
            clause_idx = -1
            for _ in range(10):
                ci_try = rng.randrange(m)
                if clause_unsat[ci_try]:
                    clause_idx = ci_try
                    break
            
            if clause_idx == -1:
                # Fallback: scan
                unsat_list = [ci for ci in range(m) if clause_unsat[ci]]
                if not unsat_list:
                    return assignment
                clause_idx = rng.choice(unsat_list)
            
            clause = clause_vars[clause_idx]
            
            # Choose literal to flip
            noise = initial_noise * (noise_decay ** step)
            noise = max(noise, 0.05)
            
            if rng.random() < noise:
                # Random literal
                lit_idx = rng.randrange(len(clause))
            else:
                # Weighted by clause weight: prefer variables that appear in many unsat clauses
                # Simple: pick literal whose variable appears in most unsat clauses
                weights = []
                for var, pos in clause:
                    # Count how many unsat clauses this variable appears in
                    w = 0
                    for ci2, pos2 in var_occurs[var]:
                        if clause_unsat[ci2]:
                            w += 1
                    weights.append(w + 1)
                
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                lit_idx = 0
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        lit_idx = i
                        break
            
            var, pos = clause[lit_idx]
            old_val = assignment[var]
            new_val = not old_val
            assignment[var] = new_val
            
            # Update clause_false_count for all clauses containing this variable
            for ci2, pos2 in var_occurs[var]:
                # Before flip: literal was (old_val == pos2) ? true : false
                was_true = (old_val == pos2)
                is_true = (new_val == pos2)
                
                if was_true and not is_true:
                    # Literal became false
                    clause_false_count[ci2] += 1
                    if clause_false_count[ci2] == len(clause_vars[ci2]):
                        clause_unsat[ci2] = 1
                        total_unsat += 1
                elif not was_true and is_true:
                    # Literal became true
                    clause_false_count[ci2] -= 1
                    if clause_false_count[ci2] == 0:
                        clause_unsat[ci2] = 0
                        total_unsat -= 1
            
            if total_unsat < best_unsat_count:
                best_unsat_count = total_unsat
                best_assignment = assignment[:]
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: return last assignment
    return assignment
# EVOLVE-BLOCK-END
