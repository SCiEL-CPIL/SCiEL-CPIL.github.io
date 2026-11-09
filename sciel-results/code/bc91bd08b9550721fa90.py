"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    """Satisfying assignment for a locally generated planted satisfiable 3-CNF.

    instance: dict with 'n' (int) and 'clauses' (list of lists of signed 1-based ints).
    Returns: list of booleans (length n) such that every clause is satisfied.
    """
    n = instance["n"]
    clauses = instance["clauses"]

    start = time.time()
    deadline = start + 295.0

    rng = random.Random(42)

    m = len(clauses)
    lit_pos = np.empty((m, 3), dtype=np.int64)
    lit_sign = np.empty((m, 3), dtype=np.int8)
    for i, cl in enumerate(clauses):
        for j, l in enumerate(cl):
            lit_pos[i, j] = abs(l) - 1
            lit_sign[i, j] = 1 if l > 0 else -1

    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = lit_pos[i, j]
            var_to_clauses[v].append(i)

    def build_value(assignment):
        return np.where(np.asarray(assignment, dtype=bool), 1, -1).astype(np.int8)

    def check_sat(value):
        vals = value[lit_pos]
        return np.any(lit_sign * vals == 1, axis=1)

    # Initial random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    value = build_value(assignment)
    unsat_mask = ~check_sat(value)
    cur_unsat = int(np.count_nonzero(unsat_mask))

    if cur_unsat == 0:
        return assignment

    best_assignment = assignment[:]
    best_unsat = cur_unsat

    # WalkSAT with clause-weighted heuristic
    # For each variable in an unsat clause, compute:
    #   score[v] = sum over all clauses k containing v of:
    #     +1 if k is unsat (flipping might fix it)
    #     -1 if k is sat and v's literal is the only true one in k (flipping would break it)
    # This is more accurate than just counting unsat clauses.

    def compute_scores(value, unsat_mask):
        """Compute score for each variable: net benefit of flipping it."""
        scores = np.zeros(n, dtype=np.int64)
        
        # For each clause, determine if it's unsat or sat
        # and if sat, whether each literal is the only true one
        
        vals = value[lit_pos]  # (m, 3)
        lit_vals = lit_sign * vals  # +1 if true, -1 if false
        true_count = np.sum(lit_vals == 1, axis=1)
        
        for i in range(m):
            if unsat_mask[i]:
                # Unsat clause: all literals false
                # Flipping any variable in this clause might fix it
                for j in range(3):
                    v = lit_pos[i, j]
                    scores[v] += 1
            else:
                # Sat clause
                if true_count[i] == 1:
                    # Only one true literal: flipping that variable would break the clause
                    only_j = np.where(lit_vals[i] == 1)[0][0]
                    v = lit_pos[i, only_j]
                    scores[v] -= 1
                # If true_count > 1, flipping any one variable won't break the clause
                # (at least one other true literal remains)
        
        return scores

    scores = compute_scores(value, unsat_mask)

    max_total_steps = 2000000
    noise = 0.3

    while time.time() < deadline:
        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        value = build_value(assignment)
        unsat_mask = ~check_sat(value)
        cur_unsat = int(np.count_nonzero(unsat_mask))

        if cur_unsat == 0:
            return assignment

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        # Occasionally restart from best
        if best_unsat <= 3 and rng.random() < 0.3:
            assignment = best_assignment[:]
            value = build_value(assignment)
            unsat_mask = ~check_sat(value)
            cur_unsat = best_unsat

        scores = compute_scores(value, unsat_mask)

        for _ in range(max_total_steps):
            if time.time() >= deadline:
                break

            if cur_unsat == 0:
                return assignment

            # Pick a random unsatisfied clause
            unsat_indices = np.nonzero(unsat_mask)[0]
            ci = rng.choice(unsat_indices.tolist())

            # For each of the 3 literals in this clause, get its score
            clause_vars = lit_pos[ci]
            clause_scores = scores[clause_vars]

            # Choose literal with max score + noise
            if rng.random() < noise:
                j = rng.randrange(3)
            else:
                max_s = clause_scores.max()
                candidates = np.nonzero(clause_scores == max_s)[0]
                j = rng.choice(candidates.tolist())

            v = clause_vars[j]
            old = value[v]
            new = -old
            value[v] = new
            assignment[v] = new > 0

            # Update unsat_mask incrementally
            for k in var_to_clauses[v]:
                sat = np.any(lit_sign[k] * value[lit_pos[k]] == 1)
                if sat:
                    if unsat_mask[k]:
                        cur_unsat -= 1
                        unsat_mask[k] = False
                else:
                    if not unsat_mask[k]:
                        cur_unsat += 1
                        unsat_mask[k] = True

            # Recompute scores (this is O(m*3) per step, which might be slow)
            # Let's only recompute every few steps
            # For now, recompute every step for correctness
            scores = compute_scores(value, unsat_mask)

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
