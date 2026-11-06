"""Improved WalkSAT with weighted clauses, stagnation detection, and time-bounded search."""
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Deadline guard: 330s wall-clock limit, use 325s as safety margin
    deadline = time.time() + 325.0
    
    # Pre-compute clause structure for speed
    m = len(clauses)
    # For each variable, store list of (clause_index, literal_sign) where sign is +1 if literal is var, -1 if ~var
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[var_idx].append((ci, sign))
    
    # Clause weights
    weights = [1.0] * m
    
    # Best assignment found so far (by number of unsat clauses)
    def initial_assignment():
        return [rng.random() < 0.5 for _ in range(n)]
    
    def count_unsat(assignment):
        """Count unsatisfied clauses efficiently."""
        unsat_count = 0
        for ci, clause in enumerate(clauses):
            sat = False
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        return unsat_count
    
    def get_unsat_clauses(assignment):
        """Return list of indices of unsatisfied clauses."""
        unsat = []
        for ci, clause in enumerate(clauses):
            sat = False
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat = True
                    break
            if not sat:
                unsat.append(ci)
        return unsat
    
    def flip_score(assignment, var_idx, var_clauses_list):
        """Compute the net change in satisfied clauses if we flip var_idx."""
        current_sat = 0
        for ci, sign in var_clauses_list:
            clause = clauses[ci]
            # Check if this variable currently satisfies the clause
            lit_val = assignment[var_idx]
            lit_satisfies = (lit_val == (sign == 1))
            if lit_satisfies:
                current_sat += 1
            # Check if the clause is currently satisfied by another literal
            clause_sat_by_other = False
            for lit in clause:
                if abs(lit) - 1 != var_idx:
                    if assignment[abs(lit) - 1] == (lit > 0):
                        clause_sat_by_other = True
                        break
            if lit_satisfies and not clause_sat_by_other:
                # Flipping would make this clause unsat (lose)
                current_sat -= 1  # net change contribution: -1
            elif not lit_satisfies and not clause_sat_by_other:
                # Flipping would satisfy this clause (gain)
                current_sat += 1  # net change contribution: +1
            # else: no change
        return current_sat
    
    # Main search loop with restarts
    best_assignment = None
    best_unsat = m + 1
    
    max_restarts = 200
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # New random assignment
        assignment = initial_assignment()
        unsat_count = count_unsat(assignment)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        # Reset weights with some noise
        for i in range(m):
            weights[i] = 1.0 + rng.random() * 0.5
        
        # Track stagnation
        last_improvement_step = 0
        stagnation_threshold = 200
        current_min_unsat = unsat_count
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Check if solved
            if unsat_count == 0:
                return assignment
            
            # Update best
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
                last_improvement_step = step
            
            # Stagnation detection: boost weights of unsat clauses
            if step - last_improvement_step > stagnation_threshold:
                unsat = get_unsat_clauses(assignment)
                for ci in unsat:
                    weights[ci] *= 1.1
                # Decay all weights slightly
                for i in range(m):
                    weights[i] *= 0.95
                last_improvement_step = step
            
            # Get unsat clauses
            unsat = get_unsat_clauses(assignment)
            if not unsat:
                return assignment
            
            # Choose clause with weighted random selection
            # Use weighted sampling: pick clause proportional to weight
            total_weight = sum(weights[ci] for ci in unsat)
            if total_weight <= 0:
                clause_idx = rng.choice(unsat)
            else:
                r = rng.random() * total_weight
                cum = 0
                clause_idx = unsat[-1]
                for ci in unsat:
                    cum += weights[ci]
                    if r <= cum:
                        clause_idx = ci
                        break
            
            clause = clauses[clause_idx]
            
            # Choose literal to flip: prefer literals that are currently false (in unsat clause)
            # with probability based on weights, and prefer flipping variables that improve most
            # Simple approach: pick a literal, then decide flip based on score
            # Better: compute flip scores for all literals in clause, pick best with some randomness
            
            flip_scores = []
            for lit in clause:
                var_idx = abs(lit) - 1
                score = flip_score(assignment, var_idx, var_clauses[var_idx])
                flip_scores.append((score, var_idx, lit))
            
            # Pick the best flip with some probability, or random with some probability
            flip_scores.sort(key=lambda x: -x[0])
            
            # With probability 0.7, pick the best; otherwise pick randomly
            if rng.random() < 0.7:
                chosen = flip_scores[0]
            else:
                chosen = rng.choice(flip_scores)
            
            var_idx = chosen[1]
            assignment[var_idx] = not assignment[var_idx]
            
            # Recompute unsat count incrementally would be faster, but for simplicity recompute
            # Actually, let's recompute only the affected clauses
            # For now, full recompute is simpler and still fast enough for 150 vars
            new_unsat = 0
            for ci, clause_c in enumerate(clauses):
                sat = False
                for lit in clause_c:
                    if assignment[abs(lit) - 1] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    new_unsat += 1
            unsat_count = new_unsat
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        return initial_assignment()
