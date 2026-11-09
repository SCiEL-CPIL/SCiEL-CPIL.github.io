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

    rng = random.Random(0)

    m = len(clauses)

    # Precompute literal arrays
    lit_pos = np.empty((m, 3), dtype=np.int64)
    lit_sign = np.empty((m, 3), dtype=np.int8)
    for i, cl in enumerate(clauses):
        for j, l in enumerate(cl):
            lit_pos[i, j] = abs(l) - 1
            lit_sign[i, j] = 1 if l > 0 else -1

    # Precompute variable -> clause indices and signs
    var_to_clauses = [[] for _ in range(n)]
    var_to_clause_signs = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = lit_pos[i, j]
            var_to_clauses[v].append(i)
            var_to_clause_signs[v].append(lit_sign[i, j])

    var_to_clauses_arr = [np.array(lst, dtype=np.int64) for lst in var_to_clauses]
    var_to_clause_signs_arr = [np.array(lst, dtype=np.int8) for lst in var_to_clause_signs]

    def build_value(assignment):
        return np.where(np.asarray(assignment, dtype=bool), 1, -1).astype(np.int8)

    def calc_sat_counts(value):
        lit_vals = lit_sign * value[lit_pos]
        return np.sum(lit_vals == 1, axis=1).astype(np.int32)

    # Fast initial random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    value = build_value(assignment)
    sat_counts = calc_sat_counts(value)
    unsat_mask = (sat_counts == 0)
    cur_unsat = int(np.count_nonzero(unsat_mask))

    if cur_unsat == 0:
        return assignment

    best_assignment = None
    best_unsat = m + 1

    max_total_steps = 2000000
    noise = 0.35
    dls_threshold = 5000  # steps without improvement before DLS
    dls_size = 30  # number of clauses to pick for DLS flip

    def dls_escape():
        """DLS: flip variables in a random subset of unsat clauses."""
        nonlocal assignment, value, sat_counts, unsat_mask, cur_unsat
        unsat_indices = np.nonzero(unsat_mask)[0]
        if len(unsat_indices) == 0:
            return
        # Pick a random subset of unsat clauses
        k = min(dls_size, len(unsat_indices))
        chosen = rng.sample(unsat_indices.tolist(), k)
        # Collect all variables in these clauses
        vars_to_flip = set()
        for ci in chosen:
            for j in range(3):
                vars_to_flip.add(lit_pos[ci, j])
        # Flip each variable
        for v in vars_to_flip:
            old_val = value[v]
            new_val = -old_val
            value[v] = new_val
            assignment[v] = new_val > 0
            # Update sat_counts incrementally
            indices = var_to_clauses_arr[v]
            signs = var_to_clause_signs_arr[v]
            old_contribs = (signs * old_val == 1).astype(np.int32)
            new_contribs = (signs * new_val == 1).astype(np.int32)
            deltas = new_contribs - old_contribs
            old_counts = sat_counts[indices].copy()
            sat_counts[indices] += deltas
            new_counts = sat_counts[indices]
            became_unsat = (old_counts > 0) & (new_counts == 0)
            became_sat = (old_counts == 0) & (new_counts > 0)
            cur_unsat += int(np.sum(became_unsat)) - int(np.sum(became_sat))
            unsat_mask[indices] = (new_counts == 0)

    def polish_pass():
        """Final polish: try to fix remaining unsat clauses via targeted flips."""
        nonlocal assignment, value, sat_counts, unsat_mask, cur_unsat
        for _ in range(500):
            if time.time() >= deadline:
                break
            if cur_unsat == 0:
                return
            unsat_indices = np.nonzero(unsat_mask)[0]
            if len(unsat_indices) == 0:
                return
            ci = rng.choice(unsat_indices.tolist())
            # Try each literal in the clause
            best_j = -1
            best_gain = -1
            for j in range(3):
                v = lit_pos[ci, j]
                indices = var_to_clauses_arr[v]
                signs = var_to_clause_signs_arr[v]
                # Simulate flip
                old_val = value[v]
                new_val = -old_val
                old_contribs = (signs * old_val == 1).astype(np.int32)
                new_contribs = (signs * new_val == 1).astype(np.int32)
                deltas = new_contribs - old_contribs
                old_counts = sat_counts[indices]
                new_counts = old_counts + deltas
                became_unsat = int(np.sum((old_counts > 0) & (new_counts == 0)))
                became_sat = int(np.sum((old_counts == 0) & (new_counts > 0)))
                gain = became_sat - became_unsat
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j >= 0 and best_gain > 0:
                v = lit_pos[ci, best_j]
                old_val = value[v]
                new_val = -old_val
                value[v] = new_val
                assignment[v] = new_val > 0
                indices = var_to_clauses_arr[v]
                signs = var_to_clause_signs_arr[v]
                old_contribs = (signs * old_val == 1).astype(np.int32)
                new_contribs = (signs * new_val == 1).astype(np.int32)
                deltas = new_contribs - old_contribs
                old_counts = sat_counts[indices].copy()
                sat_counts[indices] += deltas
                new_counts = sat_counts[indices]
                became_unsat = (old_counts > 0) & (new_counts == 0)
                became_sat = (old_counts == 0) & (new_counts > 0)
                cur_unsat += int(np.sum(became_unsat)) - int(np.sum(became_sat))
                unsat_mask[indices] = (new_counts == 0)
            else:
                break

    while time.time() < deadline:
        # Random restart assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        value = build_value(assignment)
        sat_counts = calc_sat_counts(value)
        unsat_mask = (sat_counts == 0)
        cur_unsat = int(np.count_nonzero(unsat_mask))

        if cur_unsat == 0:
            return assignment

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        # Light annealing: occasionally restart from best
        if best_unsat <= 2 and rng.random() < 0.3:
            assignment = best_assignment[:]
            value = build_value(assignment)
            sat_counts = calc_sat_counts(value)
            unsat_mask = (sat_counts == 0)
            cur_unsat = best_unsat

        steps_since_improve = 0
        for _ in range(max_total_steps):
            if time.time() >= deadline:
                break

            if cur_unsat == 0:
                return assignment

            # Pick a random unsatisfied clause
            unsat_indices = np.nonzero(unsat_mask)[0]
            if len(unsat_indices) == 0:
                return assignment
            ci = rng.choice(unsat_indices.tolist())

            # Compute gains for the 3 variables in clause ci
            gains = np.zeros(3, dtype=np.int64)
            for j in range(3):
                v = lit_pos[ci, j]
                indices = var_to_clauses_arr[v]
                count = np.sum(unsat_mask[indices]) - 1
                gains[j] = count

            # Choose literal with max gain + noise
            if rng.random() < noise:
                j = rng.randrange(3)
            else:
                maxg = gains.max()
                candidates = np.nonzero(gains == maxg)[0]
                j = rng.choice(candidates.tolist())

            v = lit_pos[ci, j]

            # Update value and assignment
            old_val = value[v]
            new_val = -old_val
            value[v] = new_val
            assignment[v] = new_val > 0

            # Vectorized update of sat_counts and unsat_mask
            indices = var_to_clauses_arr[v]
            signs = var_to_clause_signs_arr[v]

            old_contribs = (signs * old_val == 1).astype(np.int32)
            new_contribs = (signs * new_val == 1).astype(np.int32)
            deltas = new_contribs - old_contribs

            old_counts = sat_counts[indices].copy()
            sat_counts[indices] += deltas
            new_counts = sat_counts[indices]

            became_unsat = (old_counts > 0) & (new_counts == 0)
            became_sat = (old_counts == 0) & (new_counts > 0)
            cur_unsat += int(np.sum(became_unsat)) - int(np.sum(became_sat))
            unsat_mask[indices] = (new_counts == 0)

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
                steps_since_improve = 0
            else:
                steps_since_improve += 1

            # DLS escape when stuck
            if steps_since_improve > dls_threshold and cur_unsat > 0:
                dls_escape()
                steps_since_improve = 0

                if cur_unsat == 0:
                    return assignment
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]

        # End of inner loop: try polish if we're close
        if cur_unsat <= 5:
            polish_pass()
            if cur_unsat == 0:
                return assignment

    # Return best found
    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
