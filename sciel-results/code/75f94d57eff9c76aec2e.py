"""Improved 3-SAT solver using vectorized WalkSAT with impact-based literal selection."""

import random
import time
import numpy as np

# Module-level state for time management
_start_time = time.time()
# Assume 300s total limit, leave 10s buffer for safety/overhead
_DEADLINE = _start_time + 285.0


def solve(instance):
    """
    Solves a 3-SAT instance.
    Returns a list of booleans representing the assignment.
    """
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # Build matrix representation:
    # lit_matrix[m][3] = variable indices (0-based), -1 for padding
    # sign_matrix[m][3] = +1 for positive literal, -1 for negative
    lit_matrix = np.full((m, 3), -1, dtype=np.int32)
    sign_matrix = np.zeros((m, 3), dtype=np.int8)

    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            lit_matrix[i, j] = abs(lit) - 1
            sign_matrix[i, j] = 1 if lit > 0 else -1

    # Build variable-to-clause incidence for fast impact computation
    # For each variable, store which clauses contain it and at what position
    var_clause_info = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = lit_matrix[i, j]
            if v >= 0:
                var_clause_info[v].append((i, j))

    # Precompute clause count per variable (how many clauses each var appears in)
    var_clause_count = np.array([len(info) for info in var_clause_info], dtype=np.int32)

    def compute_state(assignment_np):
        """
        Compute lit_true_counts and unsat_count from numpy assignment array.
        assignment_np: shape (n,), bool array
        Returns: lit_true_counts (m,), unsat_count (int)
        """
        # For each clause, count how many literals are true
        # lit_true = (assignment[var] == (sign > 0))
        # Expand: assignment[lit_matrix] gives values for each literal position
        expanded = assignment_np[lit_matrix]  # shape (m, 3), -1 positions will be error
        # Handle -1 positions: set them to False
        expanded = np.where(lit_matrix >= 0, expanded, False)
        # A literal is true if assignment[var] matches sign
        # sign > 0 means we want assignment[var] == True
        # sign < 0 means we want assignment[var] == False
        lit_true = np.where(sign_matrix > 0, expanded, ~expanded)
        # Only count valid positions
        lit_true = np.where(lit_matrix >= 0, lit_true, False)
        lit_true_counts = lit_true.sum(axis=1)
        unsat_count = int((lit_true_counts == 0).sum())
        return lit_true_counts, unsat_count

    # Initialize random assignment
    rng = np.random.default_rng()
    assignment = rng.random(n) < 0.5
    lit_true_counts, unsat_count = compute_state(assignment)

    # Best solution tracking
    best_assignment = assignment.copy()
    best_unsat_count = unsat_count

    if unsat_count == 0:
        return best_assignment.tolist()

    # WalkSAT with restarts
    max_restarts = 300
    base_steps = 2000
    flip_threshold = 0.30  # 30% random, 70% best (more exploitation)

    for restart in range(max_restarts):
        # Check time
        if time.time() > _DEADLINE - 1.0:
            return best_assignment.tolist()

        # Reset to random assignment for each restart
        assignment = rng.random(n) < 0.5
        lit_true_counts, unsat_count = compute_state(assignment)

        if unsat_count == 0:
            return assignment.tolist()

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()

        steps = base_steps

        for step in range(steps):
            # Check time every 50 steps
            if step % 50 == 0 and time.time() > _DEADLINE - 1.0:
                return best_assignment.tolist()

            if unsat_count == 0:
                return assignment.tolist()

            # Find unsatisfied clause indices
            unsat_mask = (lit_true_counts == 0)
            unsat_indices = np.where(unsat_mask)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()

            # Pick random unsatisfied clause
            clause_idx = int(rng.choice(unsat_indices))

            # Get the 3 literals in this clause
            clause_vars = lit_matrix[clause_idx]
            clause_signs = sign_matrix[clause_idx]

            # Decide: random flip or best flip
            if rng.random() < flip_threshold:
                # Random flip: pick random variable from clause
                var_idx = int(clause_vars[rng.integers(0, 3)])
            else:
                # Best flip: compute delta for each of the 3 variables
                # Delta = change in unsat_count if we flip this variable
                # For each variable v in the clause:
                #   For each clause c containing v:
                #     if v's literal in c was true: new_count = old_count - 1
                #     else: new_count = old_count + 1
                #     delta += 1 if (old_sat and not new_sat) else (-1 if (not old_sat and new_sat) else 0)
                best_var = -1
                best_delta = float('inf')

                for j in range(3):
                    v = int(clause_vars[j])
                    delta = 0
                    for c_idx, pos in var_clause_info[v]:
                        old_cnt = lit_true_counts[c_idx]
                        old_sat = old_cnt > 0

                        # Check if v's literal in clause c_idx was true
                        s = sign_matrix[c_idx, pos]
                        if s > 0:
                            lit_was_true = assignment[v]
                        else:
                            lit_was_true = not assignment[v]

                        if lit_was_true:
                            new_cnt = old_cnt - 1
                        else:
                            new_cnt = old_cnt + 1

                        new_sat = new_cnt > 0

                        if old_sat and not new_sat:
                            delta += 1
                        elif not old_sat and new_sat:
                            delta -= 1

                    if delta < best_delta:
                        best_delta = delta
                        best_var = v
                    elif delta == best_delta and rng.random() < 0.5:
                        best_var = v

                var_idx = best_var

            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]

            # Update clause satisfaction using lit_true_counts
            for c_idx, pos in var_clause_info[var_idx]:
                s = sign_matrix[c_idx, pos]
                if s > 0:
                    # Positive literal: true when assignment[var] is True
                    if assignment[var_idx]:
                        lit_true_counts[c_idx] += 1
                        if lit_true_counts[c_idx] == 1:
                            unsat_count -= 1
                    else:
                        lit_true_counts[c_idx] -= 1
                        if lit_true_counts[c_idx] == 0:
                            unsat_count += 1
                else:
                    # Negative literal: true when assignment[var] is False
                    if not assignment[var_idx]:
                        lit_true_counts[c_idx] += 1
                        if lit_true_counts[c_idx] == 1:
                            unsat_count -= 1
                    else:
                        lit_true_counts[c_idx] -= 1
                        if lit_true_counts[c_idx] == 0:
                            unsat_count += 1

            if unsat_count == 0:
                return assignment.tolist()

            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()

    return best_assignment.tolist()
