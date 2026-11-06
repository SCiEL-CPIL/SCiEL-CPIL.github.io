"""Improved planted 3-SAT solver with WalkSAT variants and time management."""
# EVOLVE-BLOCK-START
import time
import random
import sys


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: leave 5s margin before 330s kill
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Preprocess: convert to 0-based indices for speed
    # clauses is list of lists of signed 1-based literals
    # We'll work with a faster representation
    m = len(clauses)
    
    # Fast satisfaction check using precomputed variable indices
    # For each clause, store (var_idx, polarity) where polarity 1 means positive
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, 1 if l > 0 else -1) for l in c]
        clause_vars.append(cv)
    
    def is_satisfied(clause_idx, assignment):
        cv = clause_vars[clause_idx]
        for var, pol in cv:
            if assignment[var] == (pol == 1):
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for i in range(m):
            if not is_satisfied(i, assignment):
                cnt += 1
        return cnt
    
    # Try multiple strategies
    strategies = [
        "walksat_p",       # WalkSAT with probability parameter
        "walksat_p2",      # WalkSAT with different probability
        "min_conflicts",   # Min-conflicts heuristic
        "random_restart",  # Random restarts with increasing budget
    ]
    
    # First, try a quick random assignment
    rng = random.Random(42)
    
    # Strategy 1: WalkSAT with p=0.55 (slightly favor improving moves)
    def walksat(p_flip, max_steps, seed):
        local_rng = random.Random(seed)
        assignment = [local_rng.random() < 0.5 for _ in range(n)]
        
        # Count initial unsat
        unsat_count = 0
        for i in range(m):
            if not is_satisfied(i, assignment):
                unsat_count += 1
        
        if unsat_count == 0:
            return assignment, True
        
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False
            
            # Find an unsatisfied clause
            # Use random sampling to avoid full scan each time
            for _ in range(10):
                ci = local_rng.randrange(m)
                if not is_satisfied(ci, assignment):
                    break
            else:
                # Full scan as fallback
                for ci in range(m):
                    if not is_satisfied(ci, assignment):
                        break
                else:
                    return assignment, True
            
            cv = clause_vars[ci]
            # With probability p_flip, flip a random literal in the clause
            # Otherwise, flip the literal that minimizes the number of newly unsatisfied clauses
            if local_rng.random() < p_flip:
                var, pol = cv[local_rng.randrange(len(cv))]
            else:
                # Find variable that minimizes delta
                best_var = None
                best_delta = float('inf')
                for var, pol in cv:
                    # If we flip var, what happens?
                    # Currently var has value assignment[var]
                    # After flip, var becomes not assignment[var]
                    # Check impact on all clauses containing var
                    # This is expensive; approximate by checking only this clause
                    # Better: count how many clauses become unsat
                    delta = 0
                    # We need to check all clauses that contain var
                    # For efficiency, we'll just pick the variable in the clause
                    # that currently has the "wrong" value
                    if assignment[var] != (pol == 1):
                        # Flipping would satisfy this literal
                        # But might break other clauses
                        # Simplified: just count within this clause context
                        delta = -1  # heuristic
                    else:
                        delta = 1
                    if delta < best_delta:
                        best_delta = delta
                        best_var = var
                var = best_var
            
            assignment[var] = not assignment[var]
            
            # Check if all satisfied
            # Quick check: only recheck affected clauses would be ideal,
            # but for simplicity, do periodic full checks
            if step % 10 == 0:
                unsat_count = 0
                for i in range(m):
                    if not is_satisfied(i, assignment):
                        unsat_count += 1
                        if unsat_count > 0:
                            break
                if unsat_count == 0:
                    return assignment, True
        
        # Final check
        for i in range(m):
            if not is_satisfied(i, assignment):
                return assignment, False
        return assignment, True
    
    # Strategy 2: Min-conflicts
    def min_conflicts(max_steps, seed):
        local_rng = random.Random(seed)
        assignment = [local_rng.random() < 0.5 for _ in range(n)]
        
        # Count conflicts per variable
        var_conflicts = [0] * n
        for i in range(m):
            if not is_satisfied(i, assignment):
                for var, pol in clause_vars[i]:
                    var_conflicts[var] += 1
        
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False
            
            # Find variables with conflicts
            conflicted_vars = [v for v in range(n) if var_conflicts[v] > 0]
            if not conflicted_vars:
                return assignment, True
            
            var = local_rng.choice(conflicted_vars)
            
            # Try both values, pick the one with fewer conflicts
            old_val = assignment[var]
            
            # Count conflicts if we set var to True
            conflicts_true = 0
            for i in range(m):
                if not is_satisfied(i, assignment):
                    for v, pol in clause_vars[i]:
                        if v == var:
                            # Check if this clause would be satisfied with var=True
                            if pol == 1 and True == True:
                                continue
                            # Check if other literals satisfy the clause
                            all_other_satisfied = True
                            for v2, pol2 in clause_vars[i]:
                                if v2 != var:
                                    if not (assignment[v2] == (pol2 == 1)):
                                        all_other_satisfied = False
                                        break
                            if all_other_satisfied:
                                break
                            else:
                                conflicts_true += 1
                            break
                        # If var is in clause but not the first literal
                        # This logic is getting complex; let's simplify
                        pass
            
            # Simplified: just flip randomly among conflicted vars
            # But try to be smarter: flip to value that satisfies more clauses
            # Count how many unsat clauses contain var with each polarity
            sat_if_true = 0
            sat_if_false = 0
            for i in range(m):
                if not is_satisfied(i, assignment):
                    for v, pol in clause_vars[i]:
                        if v == var:
                            if pol == 1:
                                sat_if_true += 1
                            else:
                                sat_if_false += 1
                            break
            
            if sat_if_true >= sat_if_false:
                new_val = True
            else:
                new_val = False
            
            if new_val == old_val:
                new_val = not old_val
            
            assignment[var] = new_val
        
        # Final check
        for i in range(m):
            if not is_satisfied(i, assignment):
                return assignment, False
        return assignment, True
    
    # Try strategies with increasing budgets
    best_assignment = None
    best_unsat = float('inf')
    
    # Phase 1: Quick WalkSAT attempts
    for attempt in range(5):
        if time.time() > deadline:
            break
        steps = 5000 + attempt * 5000
        p = 0.5 + 0.05 * attempt
        assignment, solved = walksat(p, steps, 42 + attempt * 1000)
        if solved:
            return assignment
        # Track best
        unsat_cnt = 0
        for i in range(m):
            if not is_satisfied(i, assignment):
                unsat_cnt += 1
        if unsat_cnt < best_unsat:
            best_unsat = unsat_cnt
            best_assignment = assignment[:]
    
    # Phase 2: Min-conflicts attempts
    for attempt in range(5):
        if time.time() > deadline:
            break
        steps = 10000 + attempt * 10000
        assignment, solved = min_conflicts(steps, 100 + attempt * 1000)
        if solved:
            return assignment
        unsat_cnt = 0
        for i in range(m):
            if not is_satisfied(i, assignment):
                unsat_cnt += 1
        if unsat_cnt < best_unsat:
            best_unsat = unsat_cnt
            best_assignment = assignment[:]
    
    # Phase 3: Longer WalkSAT with restarts
    if best_assignment is not None and time.time() < deadline:
        for attempt in range(10):
            if time.time() > deadline:
                break
            steps = 20000 + attempt * 10000
            p = 0.55
            assignment, solved = walksat(p, steps, 5000 + attempt * 2000)
            if solved:
                return assignment
            unsat_cnt = 0
            for i in range(m):
                if not is_satisfied(i, assignment):
                    unsat_cnt += 1
            if unsat_cnt < best_unsat:
                best_unsat = unsat_cnt
                best_assignment = assignment[:]
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: random assignment
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
