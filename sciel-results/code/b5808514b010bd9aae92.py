"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0  # safety margin before 330s kill

    # Precompute clause data
    m = len(clauses)
    # For each clause, store literals and variable indices
    clause_vars = []  # list of list of var indices (0-based)
    clause_polarity = []  # list of list of bool (True if positive literal)
    for c in clauses:
        vars_in_c = []
        pols_in_c = []
        for lit in c:
            var_idx = abs(lit) - 1
            vars_in_c.append(var_idx)
            pols_in_c.append(lit > 0)
        clause_vars.append(vars_in_c)
        clause_polarity.append(pols_in_c)

    # Build variable -> clause incidence (for fast updates)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for v in clause_vars[i]:
            var_to_clauses[v].append(i)

    def random_assignment(rng):
        return [rng.random() < 0.5 for _ in range(n)]

    def compute_unsat_count(assignment):
        count = 0
        for i in range(m):
            sat = False
            for j in range(len(clause_vars[i])):
                if assignment[clause_vars[i][j]] == clause_polarity[i][j]:
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def check_satisfied(assignment):
        for i in range(m):
            sat = False
            for j in range(len(clause_vars[i])):
                if assignment[clause_vars[i][j]] == clause_polarity[i][j]:
                    sat = True
                    break
            if not sat:
                return False
        return True

    def sa_search(rng, max_steps):
        assignment = random_assignment(rng)
        current_unsat = compute_unsat_count(assignment)
        if current_unsat == 0:
            return assignment

        T_start = 1.0
        T_end = 0.01
        # Cooling schedule: exponential
        if max_steps <= 1:
            return assignment
        cool_rate = (T_end / T_start) ** (1.0 / max_steps)

        best_assignment = assignment[:]
        best_unsat = current_unsat
        T = T_start

        for step in range(max_steps):
            if time.time() > deadline:
                break

            # Pick a random unsatisfied clause
            # For efficiency, pick random clause, check if unsat, else retry a few times
            clause_idx = rng.randrange(m)
            sat = False
            for j in range(len(clause_vars[clause_idx])):
                if assignment[clause_vars[clause_idx][j]] == clause_polarity[clause_idx][j]:
                    sat = True
                    break
            if sat:
                # Try a few more times to find an unsat clause
                for _ in range(5):
                    clause_idx = rng.randrange(m)
                    sat = False
                    for j in range(len(clause_vars[clause_idx])):
                        if assignment[clause_vars[clause_idx][j]] == clause_polarity[clause_idx][j]:
                            sat = True
                            break
                    if not sat:
                        break
                if sat:
                    # No unsat clause found easily; we might be at a solution or near it
                    # Just pick a random variable to flip
                    var_idx = rng.randrange(n)
                else:
                    pass  # clause_idx is now unsat
            else:
                pass  # clause_idx is unsat

            # Pick a variable from the clause to flip
            var_idx = clause_vars[clause_idx][rng.randrange(len(clause_vars[clause_idx]))]

            # Compute delta: change in unsat count if we flip var_idx
            old_val = assignment[var_idx]
            new_val = not old_val
            delta = 0
            for ci in var_to_clauses[var_idx]:
                # Check if this clause was satisfied before and will be unsat after, or vice versa
                # Count satisfied literals in this clause for old and new
                old_sat = 0
                new_sat = 0
                for j in range(len(clause_vars[ci])):
                    v = clause_vars[ci][j]
                    p = clause_polarity[ci][j]
                    if v == var_idx:
                        if old_val == p:
                            old_sat += 1
                        if new_val == p:
                            new_sat += 1
                    else:
                        if assignment[v] == p:
                            old_sat += 1
                            new_sat += 1
                old_is_sat = old_sat > 0
                new_is_sat = new_sat > 0
                if old_is_sat and not new_is_sat:
                    delta += 1
                elif not old_is_sat and new_is_sat:
                    delta -= 1

            # Simulated annealing acceptance
            if delta <= 0:
                assignment[var_idx] = new_val
                current_unsat += delta
            else:
                # Accept with probability exp(-delta / T)
                if T > 1e-10:
                    if rng.random() < math.exp(-delta / T):
                        assignment[var_idx] = new_val
                        current_unsat += delta

            # Update temperature
            T *= cool_rate

            # Track best
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]

            if current_unsat == 0:
                return assignment

        return best_assignment

    # Main loop with restarts
    rng = random.Random(0)
    best_assignment = None
    best_unsat = m + 1

    # Determine steps per restart based on time
    # Estimate: each step is roughly O(1) for small n=150, m~500
    # Let's do adaptive restarts
    step_budget = 5000
    max_restarts = 200

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Increase step budget slightly with restarts to allow deeper search
        steps = step_budget + restart * 500

        result = sa_search(rng, steps)
        unsat = compute_unsat_count(result)

        if unsat == 0:
            return result

        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = result

        # If we've been stuck, do a larger restart
        if restart % 10 == 9 and best_unsat > 0:
            # Try a longer SA run
            if time.time() < deadline - 10.0:
                result = sa_search(rng, steps * 3)
                unsat = compute_unsat_count(result)
                if unsat == 0:
                    return result
                if unsat < best_unsat:
                    best_unsat = unsat
                    best_assignment = result

    # Final check
    if best_assignment is not None and check_satisfied(best_assignment):
        return best_assignment
    if best_assignment is not None:
        return best_assignment
    return random_assignment(rng)

# EVOLVE-BLOCK-END
