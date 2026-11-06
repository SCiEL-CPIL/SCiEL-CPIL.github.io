# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # safety margin before 330s limit

    # Preprocessing: pure literal elimination + unit propagation
    # Represent literals as signed integers (1-based)
    # assignment: None means unassigned, True/False means assigned
    assignment = [None] * n
    # Track which literals are still present
    # We'll work with a list of clauses (lists of literals)
    remaining_clauses = [list(c) for c in clauses]
    
    # Pure literal elimination and unit propagation
    changed = True
    while changed:
        changed = False
        # Build literal occurrence map
        lit_count = {}
        for clause in remaining_clauses:
            for lit in clause:
                lit_count[lit] = lit_count.get(lit, 0) + 1
        
        # Pure literal elimination
        for i in range(n):
            pos = lit_count.get(i + 1, 0)
            neg = lit_count.get(-(i + 1), 0)
            if pos == 0 and neg > 0:
                # Literal (i+1) is pure (only appears negated)
                if assignment[i] is None:
                    assignment[i] = False
                    changed = True
            elif neg == 0 and pos > 0:
                # Literal -(i+1) is pure (only appears positive)
                if assignment[i] is None:
                    assignment[i] = True
                    changed = True
        
        if changed:
            # Rebuild clauses with assignments applied
            new_clauses = []
            for clause in remaining_clauses:
                new_clause = []
                for lit in clause:
                    idx = abs(lit) - 1
                    val = assignment[idx]
                    if val is not None:
                        if (lit > 0) == val:
                            # Clause satisfied
                            break
                        else:
                            # Literal false, remove it
                            continue
                    else:
                        new_clause.append(lit)
                else:
                    # No break, clause not satisfied
                    if len(new_clause) == 0:
                        # Empty clause - UNSAT (shouldn't happen for planted)
                        pass
                    elif len(new_clause) == 1:
                        # Unit clause - propagate
                        lit = new_clause[0]
                        idx = abs(lit) - 1
                        if assignment[idx] is None:
                            assignment[idx] = (lit > 0)
                            changed = True
                    else:
                        new_clauses.append(new_clause)
            remaining_clauses = new_clauses
            continue
        
        # Unit propagation (no pure literals changed)
        unit_found = False
        for clause in remaining_clauses:
            if len(clause) == 1:
                lit = clause[0]
                idx = abs(lit) - 1
                if assignment[idx] is None:
                    assignment[idx] = (lit > 0)
                    unit_found = True
                    break
        if unit_found:
            changed = True
            # Rebuild
            new_clauses = []
            for clause in remaining_clauses:
                new_clause = []
                for lit in clause:
                    idx = abs(lit) - 1
                    val = assignment[idx]
                    if val is not None:
                        if (lit > 0) == val:
                            break
                        else:
                            continue
                    else:
                        new_clause.append(lit)
                else:
                    if len(new_clause) == 0:
                        pass
                    elif len(new_clause) == 1:
                        lit = new_clause[0]
                        idx = abs(lit) - 1
                        if assignment[idx] is None:
                            assignment[idx] = (lit > 0)
                    else:
                        new_clauses.append(new_clause)
            remaining_clauses = new_clauses

    # If all assigned, check if satisfied
    all_assigned = all(a is not None for a in assignment)
    if all_assigned:
        # Verify
        for clause in clauses:
            if not any(assignment[abs(l)-1] == (l > 0) for l in clause):
                break
        else:
            return assignment

    # Build assignment array with defaults for unassigned
    full_assignment = [False] * n
    for i in range(n):
        full_assignment[i] = assignment[i] if assignment[i] is not None else False

    # WalkSAT with dynamic noise and restarts
    best_score = -1
    best_assign = full_assignment[:]
    
    rng = random.Random(42)
    
    # Count how many clauses are unsatisfied
    def count_unsat(assignment):
        cnt = 0
        for clause in clauses:
            if not any(assignment[abs(l)-1] == (l > 0) for l in clause):
                cnt += 1
        return cnt

    # Initial evaluation
    cur_score = count_unsat(full_assignment)
    if cur_score == 0:
        return full_assignment
    best_score = cur_score
    best_assign = full_assignment[:]

    max_restarts = 50
    steps_per_restart = 5000
    noise_start = 0.3
    noise_end = 0.1

    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize assignment: use best found, or random if first
        if restart == 0:
            assignment = full_assignment[:]
        else:
            # Start from best assignment with some random perturbation
            assignment = best_assign[:]
            # Flip some random variables
            num_flips = max(1, n // 10)
            for _ in range(num_flips):
                idx = rng.randrange(n)
                assignment[idx] = not assignment[idx]
        
        cur_score = count_unsat(assignment)
        if cur_score == 0:
            return assignment
        if cur_score < best_score:
            best_score = cur_score
            best_assign = assignment[:]
        
        # Dynamic noise: start high, decrease
        progress = restart / max_restarts
        noise = noise_start + (noise_end - noise_start) * progress
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 2:
                break
            
            # Find unsatisfied clauses
            unsat = []
            for c in clauses:
                if not any(assignment[abs(l)-1] == (l > 0) for l in c):
                    unsat.append(c)
            
            if not unsat:
                return assignment
            
            if len(unsat) < best_score:
                best_score = len(unsat)
                best_assign = assignment[:]
            
            # Choose a random unsatisfied clause
            clause = unsat[rng.randrange(len(unsat))]
            
            # Count how many literals in the clause would flip to true
            # Choose literal: with probability (1-noise) pick a literal that minimizes
            # the number of newly unsatisfied clauses (min-conflicts), otherwise random
            if rng.random() < (1 - noise):
                # Min-conflicts: pick literal that breaks fewest other clauses
                best_lit = None
                best_cost = float('inf')
                for lit in clause:
                    idx = abs(lit) - 1
                    # Cost = number of currently satisfied clauses that would become unsatisfied
                    cost = 0
                    for c in clauses:
                        # Check if this clause is currently satisfied and would become unsat
                        # by flipping lit
                        if any(assignment[abs(l)-1] == (l > 0) for l in c):
                            # Check if flipping idx would make this clause unsat
                            # The clause is satisfied; if all its satisfied literals are at idx,
                            # and we flip idx, it might become unsat
                            # More precisely: count how many satisfied literals are at idx
                            sat_count_at_idx = 0
                            for l in c:
                                if abs(l) - 1 == idx:
                                    if assignment[idx] == (l > 0):
                                        sat_count_at_idx += 1
                            # If all satisfied literals of c are at idx, flipping makes it unsat
                            total_sat = sum(1 for l in c if assignment[abs(l)-1] == (l > 0))
                            if sat_count_at_idx == total_sat and sat_count_at_idx > 0:
                                cost += 1
                    if cost < best_cost:
                        best_cost = cost
                        best_lit = lit
                assignment[abs(best_lit) - 1] = not assignment[abs(best_lit) - 1]
            else:
                # Random literal from the clause
                lit = clause[rng.randrange(len(clause))]
                assignment[abs(lit) - 1] = not assignment[abs(lit) - 1]
        
        # End of restart
        cur_score = count_unsat(assignment)
        if cur_score < best_score:
            best_score = cur_score
            best_assign = assignment[:]
        if best_score == 0:
            return best_assign

    return best_assign
# EVOLVE-BLOCK-END
