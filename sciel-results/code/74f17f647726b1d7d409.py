"""Synthetic planted 3-SAT solver (50 variables) with restarts and adaptive weights."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Precompute literal indices and signs for speed
    # lit_to_idx: 0-based index for variable
    # lit_sign: 1 if positive, -1 if negative
    lit_to_idx = [0] * (n + 1)
    lit_sign = [0] * (n + 1)
    for i in range(1, n + 1):
        lit_to_idx[i] = i - 1
        lit_sign[i] = 1
        lit_to_idx[-i] = i - 1
        lit_sign[-i] = -1

    deadline = time.time() + 290  # 10s safety margin before 300s limit

    # Weight vector for each literal (positive and negative)
    # weight[lit] tracks how often flipping this literal helped
    # We'll use a simpler approach: count how many clauses each variable appears in
    
    # Variable occurrence counts for biasing initial assignment
    var_pos_count = [0] * n  # count of positive literals
    var_neg_count = [0] * n  # count of negative literals
    for c in clauses:
        for lit in c:
            idx = abs(lit) - 1
            if lit > 0:
                var_pos_count[idx] += 1
            else:
                var_neg_count[idx] += 1

    def is_satisfied(assignment, clause):
        for lit in clause:
            idx = abs(lit) - 1
            if lit > 0:
                if assignment[idx]:
                    return True
            else:
                if not assignment[idx]:
                    return True
        return False

    def count_unsat(assignment):
        count = 0
        for c in clauses:
            for lit in c:
                idx = abs(lit) - 1
                if lit > 0:
                    if assignment[idx]:
                        break
                else:
                    if not assignment[idx]:
                        break
            else:
                count += 1
        return count

    best_assignment = None
    best_unsat_count = m + 1

    rng = random.Random(42)

    # WalkSAT with restarts and adaptive weight adjustment
    max_restarts = 500
    max_steps_per_restart = 5000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize assignment
        # Use biased initialization based on literal frequencies
        assignment = [False] * n
        for i in range(n):
            # If a variable appears more positively, bias towards True
            total = var_pos_count[i] + var_neg_count[i]
            if total > 0:
                p_true = var_pos_count[i] / total
            else:
                p_true = 0.5
            assignment[i] = rng.random() < p_true

        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        # WalkSAT with weight-based variable selection
        # Weight for each variable: how many unsatisfied clauses it can affect
        # We'll use a simpler heuristic: prefer variables in the most constrained unsat clauses
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break

            # Find unsatisfied clauses
            unsat_clauses = []
            for c in clauses:
                sat = False
                for lit in c:
                    idx = abs(lit) - 1
                    if lit > 0:
                        if assignment[idx]:
                            sat = True
                            break
                    else:
                        if not assignment[idx]:
                            sat = True
                            break
                if not sat:
                    unsat_clauses.append(c)

            if not unsat_clauses:
                return assignment

            if len(unsat_clauses) < best_unsat_count:
                best_unsat_count = len(unsat_clauses)
                best_assignment = assignment[:]

            # Select an unsatisfied clause (weighted by number of flippable variables)
            # Pick the clause with the most "flexible" variables
            clause = rng.choice(unsat_clauses)

            # Score each variable in the clause based on how many other unsat clauses it appears in
            # This is the "min-conflicts" heuristic
            best_var = None
            best_score = -1
            for lit in clause:
                var_idx = abs(lit) - 1
                # Count how many unsat clauses contain this variable
                score = 0
                for uc in unsat_clauses:
                    for ulit in uc:
                        if abs(ulit) - 1 == var_idx:
                            score += 1
                            break
                if score > best_score:
                    best_score = score
                    best_var = var_idx

            if best_var is None:
                # Fallback: random variable from clause
                lit = rng.choice(clause)
                best_var = abs(lit) - 1

            # Flip the variable
            assignment[best_var] = not assignment[best_var]

            # Update unsat count
            new_unsat_count = count_unsat(assignment)
            if new_unsat_count < best_unsat_count:
                best_unsat_count = new_unsat_count
                best_assignment = assignment[:]

            if new_unsat_count == 0:
                return assignment

    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
