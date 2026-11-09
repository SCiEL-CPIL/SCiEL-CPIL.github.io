"""Planted 3-SAT solver with optimized WalkSAT, min-conflicts, and restarts."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: 330s wall-clock, use 295s for safety
    start_time = time.time()
    deadline = start_time + 295.0

    # Preprocess: convert to 0-based, store as (var_idx, sign) where sign is +1 or -1
    # For each clause, store the list of (var, sign)
    # Also build var -> clause index mapping for efficient updates
    clause_vars = []  # list of lists of (var_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        cv = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cv.append((v, s))
            var_to_clauses[v].append(ci)
        clause_vars.append(cv)

    # State: assignment as numpy array of booleans
    # sat_count[c] = number of literals satisfied in clause c
    # We track which literals are satisfied per clause for quick updates

    def try_assignment(assignment, max_steps=50000):
        """Run WalkSAT with min-conflicts heuristic on given assignment."""
        # assignment is a list of booleans (length n)
        # Compute initial sat_count
        sat_count = np.zeros(m, dtype=np.int32)
        for c in range(m):
            for (v, s) in clause_vars[c]:
                if assignment[v] == (s == 1):
                    sat_count[c] += 1

        if np.all(sat_count > 0):
            return assignment

        # Precompute for each variable, which clauses it appears in
        # And for each (clause, var) pair, the sign
        # var_clause_sign[v] = list of (clause_idx, sign)
        var_clause_sign = [[] for _ in range(n)]
        for ci, cv in enumerate(clause_vars):
            for (v, s) in cv:
                var_clause_sign[v].append((ci, s))

        best_assignment = list(assignment)
        best_unsat = int(np.sum(sat_count == 0))

        for step in range(max_steps):
            if time.time() > deadline:
                return best_assignment

            # Find unsatisfied clauses
            unsat_mask = (sat_count == 0)
            n_unsat = int(np.sum(unsat_mask))
            if n_unsat == 0:
                return list(assignment)

            if n_unsat < best_unsat:
                best_unsat = n_unsat
                best_assignment = list(assignment)

            # Pick a random unsatisfied clause
            unsat_indices = np.where(unsat_mask)[0]
            c = int(unsat_indices[random.randrange(len(unsat_indices))])

            # Min-conflicts: pick variable that appears in most unsatisfied clauses
            # But for speed, use a weighted random selection
            # Score each variable in the clause by how many other unsat clauses it appears in
            best_var = None
            best_score = -1
            for (v, s) in clause_vars[c]:
                # Count how many unsat clauses contain v
                score = 0
                for (ci2, s2) in var_clause_sign[v]:
                    if sat_count[ci2] == 0:
                        score += 1
                # Add small random perturbation
                score += random.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_var = v

            if best_var is None:
                # Fallback: random variable in clause
                v = random.choice(clause_vars[c])[0]
            else:
                v = best_var

            # Flip variable v
            old_val = assignment[v]
            assignment[v] = not old_val

            # Update sat_count for all clauses containing v
            for (ci2, s2) in var_clause_sign[v]:
                # Before flip: literal satisfied if assignment[v] == (s2 == 1)
                was_sat = old_val == (s2 == 1)
                is_sat = (not old_val) == (s2 == 1)
                if was_sat and not is_sat:
                    sat_count[ci2] -= 1
                elif not was_sat and is_sat:
                    sat_count[ci2] += 1

        return best_assignment

    # Multiple restarts with different seeds
    best_result = None
    best_score = -1

    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(r * 7919 + 13)
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Use adaptive step budget: more steps for later restarts
        base_steps = 20000
        step_budget = base_steps + r * 5000

        result = try_assignment(assignment, max_steps=step_budget)

        # Evaluate
        unsat_count = 0
        for c in range(m):
            sat = False
            for (v, s) in clause_vars[c]:
                if result[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                unsat_count += 1

        if unsat_count == 0:
            return result

        if unsat_count < best_score:
            best_score = unsat_count
            best_result = result

        # If we've used a lot of time, reduce remaining restarts
        elapsed = time.time() - start_time
        if elapsed > 250:
            break

    # Final: try local search from best_result with more steps
    if best_result is not None and time.time() < deadline - 10:
        result = try_assignment(best_result, max_steps=100000)
        unsat_count = 0
        for c in range(m):
            sat = False
            for (v, s) in clause_vars[c]:
                if result[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        if unsat_count == 0:
            return result
        if unsat_count < best_score:
            best_result = result

    return best_result if best_result is not None else [False] * n

# EVOLVE-BLOCK-END
