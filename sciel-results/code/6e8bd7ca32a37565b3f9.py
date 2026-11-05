"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    """Simulated annealing solver for planted 3-SAT with efficient clause tracking."""
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    deadline = time.time() + 280  # Safety margin before 300s kill

    # Build variable-to-clause index for fast delta computation
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(ci)

    def initial_satisfied_count(assignment):
        """Count satisfied clauses for an assignment."""
        sat_count = 0
        for clause in clauses:
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat_count += 1
                    break
        return sat_count

    def compute_delta(var_idx, new_val, assignment, clause_sat_flags):
        """
        Compute change in satisfied clause count if we flip var_idx to new_val.
        Returns (delta_sat, affected_clauses) where delta_sat is change in #satisfied.
        """
        old_val = assignment[var_idx]
        if old_val == new_val:
            return 0, []

        delta = 0
        affected = []

        for ci in var_to_clauses[var_idx]:
            clause = clauses[ci]
            was_sat = clause_sat_flags[ci]

            # Check if clause would be satisfied after flip
            would_be_sat = False
            for lit in clause:
                v = abs(lit) - 1
                if v == var_idx:
                    if new_val == (lit > 0):
                        would_be_sat = True
                        break
                else:
                    if assignment[v] == (lit > 0):
                        would_be_sat = True
                        break

            if was_sat and not would_be_sat:
                delta -= 1
            elif not was_sat and would_be_sat:
                delta += 1

            affected.append(ci)

        return delta, affected

    # Simulated annealing with restarts
    best_assignment = None
    best_sat_count = -1
    num_restarts = 30

    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break

        rng = random.Random(42 + restart * 1000)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Initialize clause satisfaction flags
        clause_sat_flags = [False] * num_clauses
        for ci, clause in enumerate(clauses):
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    clause_sat_flags[ci] = True
                    break

        current_sat = sum(clause_sat_flags)

        if current_sat == num_clauses:
            return assignment

        if current_sat > best_sat_count:
            best_sat_count = current_sat
            best_assignment = assignment[:]

        # Simulated annealing parameters
        # Temperature schedule: exponential cooling
        t0 = 1.0
        t_min = 0.01
        # Number of steps: adapt based on remaining time
        # Estimate: ~5000 steps per second is reasonable
        remaining_time = deadline - time.time()
        max_steps = int(min(20000, max(5000, remaining_time * 5000)))
        cooling_rate = (t_min / t0) ** (1.0 / max(max_steps, 1))

        temperature = t0
        step = 0
        no_improve_steps = 0
        max_no_improve = 2000

        while step < max_steps:
            if time.time() > deadline - 5:
                break

            # Pick a random variable to flip
            var_idx = rng.randrange(n)
            new_val = not assignment[var_idx]

            delta, affected = compute_delta(var_idx, new_val, assignment, clause_sat_flags)

            if delta >= 0:
                # Accept improvement
                assignment[var_idx] = new_val
                for ci in affected:
                    clause = clauses[ci]
                    clause_sat_flags[ci] = any(
                        assignment[abs(l) - 1] == (l > 0) for l in clause
                    )
                current_sat += delta
                no_improve_steps = 0

                if current_sat == num_clauses:
                    return assignment

                if current_sat > best_sat_count:
                    best_sat_count = current_sat
                    best_assignment = assignment[:]
            else:
                # Deterioration: accept with probability exp(delta/temperature)
                # delta is negative, so delta/temperature is negative
                prob = math.exp(delta / max(temperature, 1e-10))
                if rng.random() < prob:
                    assignment[var_idx] = new_val
                    for ci in affected:
                        clause = clauses[ci]
                        clause_sat_flags[ci] = any(
                            assignment[abs(l) - 1] == (l > 0) for l in clause
                        )
                    current_sat += delta
                    no_improve_steps += 1
                else:
                    no_improve_steps += 1

            # Cool down
            temperature *= cooling_rate

            step += 1

            # Early termination if stuck for too long
            if no_improve_steps > max_no_improve and temperature < 0.1:
                break

        # If we got stuck, try a small random perturbation
        if current_sat < num_clauses and time.time() < deadline - 10:
            # Perturb a few random variables
            for _ in range(min(5, n)):
                v = rng.randrange(n)
                assignment[v] = not assignment[v]

            # Recompute clause sat flags
            for ci in range(num_clauses):
                clause = clauses[ci]
                clause_sat_flags[ci] = any(
                    assignment[abs(l) - 1] == (l > 0) for l in clause
                )
            current_sat = sum(clause_sat_flags)

            if current_sat > best_sat_count:
                best_sat_count = current_sat
                best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback
    return assignment
# EVOLVE-BLOCK-END
