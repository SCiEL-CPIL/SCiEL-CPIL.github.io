"""Improved 3-SAT solver using optimized WalkSAT with restarts and gain-based selection."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    if num_clauses == 0:
        return [False] * n
    
    # Preprocess clauses: convert to 0-based
    # Store as (var_index, sign) where sign is +1 or -1
    processed_clauses = []
    for clause in clauses:
        processed = [(abs(lit) - 1, 1 if lit > 0 else -1) for lit in clause]
        processed_clauses.append(processed)
    
    # Build variable-to-clause mapping for efficient gain computation
    # var_to_clauses[var] = list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(processed_clauses):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
    
    # Precompute clause literal arrays for faster satisfaction checks
    # For each clause, store var indices and signs as separate lists for speed
    clause_vars = []
    clause_signs = []
    for clause in processed_clauses:
        vs = [v for v, s in clause]
        ss = [s for v, s in clause]
        clause_vars.append(vs)
        clause_signs.append(ss)
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    rng = random.Random(42)
    
    # Fast satisfaction check for a single clause
    def clause_satisfied(ci, assignment):
        vs = clause_vars[ci]
        ss = clause_signs[ci]
        for i in range(len(vs)):
            v = vs[i]
            s = ss[i]
            if s == 1:
                if assignment[v]:
                    return True
            else:
                if not assignment[v]:
                    return True
        return False
    
    def count_unsat(assignment):
        count = 0
        for ci in range(num_clauses):
            if not clause_satisfied(ci, assignment):
                count += 1
        return count
    
    def get_gain(assignment, var):
        """Compute gain of flipping var: how many more clauses would be satisfied."""
        gain = 0
        for ci, sign in var_to_clauses[var]:
            vs = clause_vars[ci]
            ss = clause_signs[ci]
            # Check if clause is currently satisfied by any literal other than var
            sat_by_other = False
            for i in range(len(vs)):
                if vs[i] == var:
                    continue
                v = vs[i]
                s = ss[i]
                if s == 1:
                    if assignment[v]:
                        sat_by_other = True
                        break
                else:
                    if not assignment[v]:
                        sat_by_other = True
                        break
            
            var_val = assignment[var]
            var_satisfies = (sign == 1 and var_val) or (sign == -1 and not var_val)
            
            if sat_by_other:
                # Clause is satisfied regardless of var
                continue
            else:
                # Clause satisfaction depends on var
                if var_satisfies:
                    # Currently satisfied by var, flipping makes it unsatisfied
                    gain -= 1
                else:
                    # Currently unsatisfied, flipping makes it satisfied
                    gain += 1
        
        return gain
    
    # WalkSAT with restarts
    max_restarts = 100
    steps_per_restart = 5000
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT steps
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for ci in range(num_clauses):
                if not clause_satisfied(ci, assignment):
                    unsat_clauses.append(ci)
            
            if not unsat_clauses:
                return assignment
            
            unsat_count = len(unsat_clauses)
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            clause = processed_clauses[ci]
            
            # Choose variable to flip: with probability p, use gain; otherwise random
            if rng.random() < 0.8:  # 80% use gain-based
                best_var = None
                best_gain = -float('inf')
                for var, sign in clause:
                    gain = get_gain(assignment, var)
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                if best_var is not None:
                    assignment[best_var] = not assignment[best_var]
            else:
                var, sign = rng.choice(clause)
                assignment[var] = not assignment[var]
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
