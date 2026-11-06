"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290.0  # safety margin before 300s limit

    # Preprocess: build literal index arrays (0-based) and sign arrays
    # lit_idx[i][j] = variable index (0-based) for clause i, literal j
    # lit_sign[i][j] = +1 if positive literal, -1 if negative
    lit_idx = np.empty((m, 3), dtype=np.int32)
    lit_sign = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            lit_idx[i, j] = abs(l) - 1
            lit_sign[i, j] = 1 if l > 0 else -1

    # Initialize random assignment
    rng = np.random.RandomState(0)
    assignment = rng.random(n) < 0.5  # bool array

    # Helper: check if all clauses satisfied
    def is_satisfied():
        # assignment as int: True=1, False=0
        # A literal is satisfied if (sign==1 and var is True) or (sign==-1 and var is False)
        # i.e., sign * (2*var - 1) == 1  ->  var == (sign==1)
        # Satisfied literal: assignment[lit_idx] == (lit_sign == 1)
        sat_literals = (assignment[np.int64(lit_idx)] == (lit_sign == 1))  # (m, 3) bool
        return sat_literals.any(axis=1)  # (m,) bool: clause is satisfied

    # Check initial
    if is_satisfied().all():
        return assignment.tolist()

    # WalkSAT with vectorized unsatisfied clause detection
    # Strategy: pick a random unsatisfied clause, flip a literal in it
    # Weighted flip: prefer literals that would satisfy more other clauses

    best_assignment = assignment.copy()
    best_unsat_count = m

    # Main search loop
    step = 0
    while time.time() < deadline:
        # Find unsatisfied clauses
        sat_literals = (assignment[np.int64(lit_idx)] == (lit_sign == 1))
        clause_sat = sat_literals.any(axis=1)
        unsat_indices = np.where(~clause_sat)[0]

        if len(unsat_indices) == 0:
            return assignment.tolist()

        unsat_count = len(unsat_indices)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()

        # Pick a random unsatisfied clause
        clause_i = unsat_indices[rng.randint(len(unsat_indices))]

        # Get the 3 literals in this clause
        c_idx = lit_idx[clause_i]
        c_sign = lit_sign[clause_i]

        # For each literal, compute how many other unsatisfied clauses it appears in
        # (flipping it might satisfy more clauses)
        # Also consider: if we flip this variable, which currently-satisfied clauses become unsatisfied?

        # Simple approach: random choice among the 3 literals
        # Better: weighted by clause frequency in unsat set
        # For now, use a mix: 70% random, 30% weighted by how many unsat clauses contain the variable

        if rng.random() < 0.7:
            lit_pos = rng.randint(3)
        else:
            # Weight by count of variable in unsatisfied clauses
            weights = np.zeros(3)
            for j in range(3):
                var = c_idx[j]
                # Count how many unsat clauses contain this variable
                # This is expensive; approximate by just picking random
                weights[j] = 1.0
            # Just pick random
            lit_pos = rng.randint(3)

        # Flip the variable
        var_to_flip = c_idx[lit_pos]
        assignment[var_to_flip] = not assignment[var_to_flip]

        step += 1

        # Periodic restart if stuck
        if step % 5000 == 0:
            # Check if we're making progress
            sat_literals = (assignment[np.int64(lit_idx)] == (lit_sign == 1))
            clause_sat = sat_literals.any(axis=1)
            current_unsat = (~clause_sat).sum()
            if current_unsat >= best_unsat_count and step > 10000:
                # Restart with best assignment
                assignment = best_assignment.copy()
                step = 0

    # Return best found
    return best_assignment.tolist()
# EVOLVE-BLOCK-END
