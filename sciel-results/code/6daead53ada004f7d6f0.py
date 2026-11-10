"""Improved WalkSAT with min-conflicts, restarts, and deadline guard for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign) for speed
    # var_idx is 0-based, sign is +1 for positive, -1 for negative
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    def is_satisfied(assignment):
        """Check if all clauses are satisfied."""
        for clause in clause_data:
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for clause in clause_data:
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def walk_one(assignment, rng, max_steps, p_flip):
        """Run WalkSAT from current assignment for up to max_steps with flip probability p_flip."""
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            # Find unsatisfied clauses
            unsat_clauses = []
            for i, clause in enumerate(clause_data):
                sat = False
                for var_idx, sign in clause:
                    if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)

            if not unsat_clauses:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            clause = clause_data[ci]

            # Score each literal: count how many other unsatisfied clauses contain this variable
            # Higher count means flipping is more likely to help satisfy more clauses
            var_unsat_count = {}
            for uc in unsat_clauses:
                for var_idx, sign in clause_data[uc]:
                    var_unsat_count[var_idx] = var_unsat_count.get(var_idx, 0) + 1

            # Weighted selection with noise: combine clause frequency with random component
            # The p_flip parameter controls how much randomness is in the choice
            weights = []
            for var_idx, sign in clause:
                base_w = var_unsat_count.get(var_idx, 0) + 1
                noise = rng.random() * p_flip * 5  # add noise scaled by p_flip
                weights.append(base_w + noise)

            # Normalize and pick
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen_idx = 0
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = idx
                    break

            var_idx, sign = clause[chosen_idx]
            assignment[var_idx] = not assignment[var_idx]

        return assignment, False

    def polish(assignment, rng, max_iters=5000):
        """Final high-precision polish: try to reduce remaining unsat clauses to zero."""
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            unsat_clauses = []
            for i, clause in enumerate(clause_data):
                sat = False
                for var_idx, sign in clause:
                    if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)
            if not unsat_clauses:
                return assignment, True
            # Try flipping each variable in each unsat clause and see if it reduces total unsat
            current_unsat = len(unsat_clauses)
            best_flip = None
            best_new_unsat = current_unsat
            for uc in unsat_clauses:
                for var_idx, sign in clause_data[uc]:
                    # Try flipping
                    assignment[var_idx] = not assignment[var_idx]
                    new_unsat = 0
                    for j, cl in enumerate(clause_data):
                        sat = False
                        for vi, s in cl:
                            if (assignment[vi] and s == 1) or (not assignment[vi] and s == -1):
                                sat = True
                                break
                        if not sat:
                            new_unsat += 1
                    assignment[var_idx] = not assignment[var_idx]
                    if new_unsat < best_new_unsat:
                        best_new_unsat = new_unsat
                        best_flip = var_idx
            if best_flip is not None:
                assignment[best_flip] = not assignment[best_flip]
            else:
                break
        unsat_final = 0
        for i, clause in enumerate(clause_data):
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                unsat_final += 1
        return assignment, unsat_final == 0

    # Multi-restart strategy with varying parameters
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Parameters tuned for planted 3-SAT with 150 variables
    # With incremental tracking, each step is faster, so we can do more steps
    max_steps_per_restart = 50000
    num_restarts = 1500  # max restarts, will stop at deadline

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Vary the flip probability to explore different regimes
        p_flip = 0.2 + 0.3 * (restart % 5) / 4.0  # cycle through 0.2, 0.275, 0.35, 0.425, 0.5

        # Try to solve
        assignment, solved = walk_one(assignment, rng, max_steps_per_restart, p_flip)

        if solved:
            return assignment

        # If close to solved, try a polish pass
        uc = count_unsat(assignment)
        if uc <= 3 and time.time() < deadline - 10:
            assignment, solved = polish(assignment, rng, max_iters=1000)
            if solved:
                return assignment

        # Track best
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # Final polish on best assignment if we have time
    if best_assignment is not None and best_unsat_count <= 5 and time.time() < deadline - 5:
        best_assignment, solved = polish(best_assignment, rng, max_iters=2000)
        if solved:
            return best_assignment

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
