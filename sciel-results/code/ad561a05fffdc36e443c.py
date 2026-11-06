"""Improved 3-SAT solver for planted instances (50 vars)."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280.0  # safety margin before 300s kill

    # Preprocess: build literal -> set of clause indices, and clause literals as 0-based signed
    # Represent each clause as tuple of (var_idx, is_positive)
    clause_data = []
    for c in clauses:
        clause_data.append(tuple((abs(l) - 1, l > 0) for l in c))

    num_clauses = len(clause_data)

    # Helper to evaluate satisfaction quickly
    def is_satisfied(assignment, clause):
        for var_idx, pos in clause:
            if assignment[var_idx] == pos:
                return True
        return False

    def count_unsat(assignment):
        cnt = 0
        for clause in clause_data:
            sat = False
            for var_idx, pos in clause:
                if assignment[var_idx] == pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    # WalkSAT with restarts and adaptive noise
    def walksat(assignment, rng, max_steps):
        for step in range(max_steps):
            if time.time() > deadline:
                return None
            # Find unsatisfied clauses
            unsat = []
            for i, clause in enumerate(clause_data):
                sat = False
                for var_idx, pos in clause:
                    if assignment[var_idx] == pos:
                        sat = True
                        break
                if not sat:
                    unsat.append(i)
            if not unsat:
                return assignment
            
            # Pick random unsatisfied clause
            clause_idx = rng.choice(unsat)
            clause = clause_data[clause_idx]
            
            # Score each literal in the clause:
            # For each variable in the clause, compute how many other unsat clauses
            # would be satisfied if we flip this variable
            best_var = None
            best_score = -1
            best_val = None
            
            for var_idx, pos in clause:
                # If we set assignment[var_idx] = pos, how many unsat clauses become sat?
                # And how many currently sat clauses become unsat?
                gain = 0
                loss = 0
                for j, other_clause in enumerate(clause_data):
                    if j == clause_idx:
                        continue
                    # Check if other_clause is currently unsat
                    other_sat = False
                    for v, p in other_clause:
                        if assignment[v] == p:
                            other_sat = True
                            break
                    
                    if not other_sat:
                        # Currently unsat. Would flipping var_idx to pos satisfy it?
                        # Only if var_idx is in other_clause with the desired polarity
                        found = False
                        for v, p in other_clause:
                            if v == var_idx:
                                if p == pos:
                                    # Setting var_idx=pos satisfies this clause
                                    gain += 1
                                # else: setting var_idx=pos doesn't help (it needs opposite)
                                found = True
                                break
                        # If var_idx not in other_clause, no effect
                    else:
                        # Currently sat. Would flipping var_idx to pos break it?
                        # Only if var_idx is the ONLY literal satisfying other_clause
                        # and var_idx appears in other_clause
                        only_sat = True
                        for v, p in other_clause:
                            if assignment[v] == p:
                                if v == var_idx:
                                    # This is the satisfying literal, and we're changing it
                                    # But wait: we're setting var_idx to pos. 
                                    # If current assignment[var_idx] != pos, then flipping changes it.
                                    # If current assignment[var_idx] == pos, no change.
                                    pass
                                else:
                                    only_sat = False
                                    break
                        # This logic is getting complex; let's simplify
                        # Actually, let's just count: if var_idx is in other_clause and 
                        # currently satisfies it, and it's the only one, then loss += 1
                        # Simpler approach:
                        pass
                
                # Simpler scoring: just count how many unsat clauses contain var_idx
                # with the polarity that would be satisfied
                for j, other_clause in enumerate(clause_data):
                    if j == clause_idx:
                        continue
                    # Is other_clause currently unsat?
                    other_sat = False
                    for v, p in other_clause:
                        if assignment[v] == p:
                            other_sat = True
                            break
                    if not other_sat:
                        for v, p in other_clause:
                            if v == var_idx and p == pos:
                                gain += 1
                                break
                
                score = gain
                if score > best_score:
                    best_score = score
                    best_var = var_idx
                    best_val = pos
            
            if best_var is None:
                # Random flip
                var_idx, pos = rng.choice(clause)
                assignment[var_idx] = pos
            else:
                assignment[best_var] = best_val
        
        return None

    # Try multiple restarts with different random seeds
    best_assignment = None
    best_unsat = float('inf')
    
    num_restarts = 50
    steps_per_restart = 5000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(12345 + restart * 7919)
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Run WalkSAT
        result = walksat(assignment, rng, steps_per_restart)
        
        if result is not None:
            # Check if actually satisfied
            if count_unsat(result) == 0:
                return result
        
        # Track best partial solution
        if result is not None:
            unsat_cnt = count_unsat(result)
            if unsat_cnt < best_unsat:
                best_unsat = unsat_cnt
                best_assignment = result[:]
    
    # If we found a satisfying assignment during search, it would have been returned.
    # Return best partial solution or a random one
    if best_assignment is not None:
        return best_assignment
    else:
        return [random.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
