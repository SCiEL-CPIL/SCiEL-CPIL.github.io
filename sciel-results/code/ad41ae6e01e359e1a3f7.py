"""Improved planted 3-SAT solver using vectorized WalkSAT with restarts and time guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280.0  # safety margin before 330s kill

    # Preprocess: convert clauses to 0-based signed indices
    # literal l -> index = abs(l)-1, sign = 1 if l>0 else -1
    m = len(clauses)
    if m == 0:
        return [True] * n

    # Build numpy arrays for clauses
    # Each clause: (var_indices, signs)
    var_idx = np.zeros((m, 3), dtype=np.int32)
    signs = np.zeros((m, 3), dtype=np.int8)
    for i, cl in enumerate(clauses):
        for j, lit in enumerate(cl):
            var_idx[i, j] = abs(lit) - 1
            signs[i, j] = 1 if lit > 0 else -1

    best_assignment = None
    best_unsat = float('inf')

    rng = np.random.default_rng(42)

    while time.time() < deadline:
        # Random initial assignment
        assignment = rng.random(n) < 0.5

        # WalkSAT with adaptive parameters
        max_steps = 50000
        p = 0.5  # probability of choosing a literal from the failing clause

        # Precompute clause satisfaction efficiently
        # For each clause, check if any literal is satisfied
        # sat[i] = True if clause i is satisfied

        # Vectorized satisfaction check
        # For each clause i and literal j: satisfied if assignment[var_idx[i,j]] == (signs[i,j] > 0)
        # We'll track unsatisfied count per clause

        # Initialize: compute which clauses are satisfied
        # For each clause, a literal is satisfied if assignment[var] matches sign
        # Let's use a different approach: track for each clause how many literals are satisfied
        # sat_count[i] = number of literals in clause i that are satisfied
        # A clause is unsatisfied if sat_count[i] == 0

        # Build lookup: for each variable, which clauses contain it and with what sign
        # This is expensive for 50 vars and potentially many clauses, but let's try direct approach

        # Direct vectorized check per step is O(m*3) which for m~1000-5000 is fine
        # Let's just do it directly but with numpy for speed

        # For a single flip of variable v:
        # - Clauses containing v: their sat_count changes
        # - We need to know, for each clause containing v, whether the literal was satisfied before and after

        # Precompute: for each variable, list of (clause_idx, sign) where it appears
        var_clauses = [[] for _ in range(n)]
        for i in range(m):
            for j in range(3):
                v = var_idx[i, j]
                var_clauses[v].append((i, signs[i, j]))

        # sat_count: number of satisfied literals per clause
        # Initialize
        assignment_arr = assignment.astype(np.int8)  # 1 for True, 0 for False
        # For each clause, count satisfied literals
        # literal satisfied if assignment[var] == (sign > 0)
        # i.e., if sign==1, assignment[var]==1; if sign==-1, assignment[var]==0
        # satisfied = (assignment[var] * sign) > 0  ... wait
        # sign==1: satisfied if assignment[var]==1 -> assignment[var]*1 == 1 > 0
        # sign==-1: satisfied if assignment[var]==0 -> assignment[var]*(-1) == 0, not > 0
        # Better: satisfied = (assignment[var] == (sign > 0))
        # In int: sign>0 is 1, sign<0 is 0. assignment is 0 or 1.
        # satisfied = (assignment[var] == (sign > 0))
        # Let's use: sat = 1 if (assignment[var] == 1 and sign == 1) or (assignment[var] == 0 and sign == -1) else 0
        # = 1 if assignment[var] * sign > 0 else 0 ... no
        # assignment=1, sign=1: 1*1=1>0 -> sat
        # assignment=0, sign=-1: 0*(-1)=0, not >0 -> but should be sat!
        # So: sat = 1 if (assignment[var] == 1 and sign == 1) or (assignment[var] == 0 and sign == -1)
        # = 1 if sign * (2*assignment[var] - 1) > 0
        # sign=1, assign=1: 1*(2-1)=1>0 sat
        # sign=1, assign=0: 1*(0-1)=-1<0 unsat
        # sign=-1, assign=1: -1*(2-1)=-1<0 unsat
        # sign=-1, assign=0: -1*(0-1)=1>0 sat
        # Good!

        # Convert assignment to int: 1 for True, 0 for False
        # sat = sign * (2*assignment - 1) > 0

        # Initial sat_count
        # For each clause i, sat_count[i] = sum over j of [signs[i,j] * (2*assignment[var_idx[i,j]] - 1) > 0]
        # Vectorized:
        val = 2 * assignment_arr.astype(np.int32) - 1  # +1 for True, -1 for False
        # For each (i,j): satisfied if signs[i,j] * val[var_idx[i,j]] > 0
        # Build matrix: signs[i,j] * val[var_idx[i,j]]
        vvals = val[var_idx]  # shape (m, 3)
        sat_matrix = signs * vvals  # shape (m, 3)
        sat_count = (sat_matrix > 0).sum(axis=1).astype(np.int32)  # shape (m,)

        unsat_clauses = np.where(sat_count == 0)[0]
        cur_unsat = len(unsat_clauses)

        if cur_unsat == 0:
            return assignment.tolist()

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment.copy()

        # WalkSAT loop
        for step in range(max_steps):
            if time.time() >= deadline:
                break

            if cur_unsat == 0:
                return assignment.tolist()

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment.copy()

            # Pick a random unsatisfied clause
            uc = unsat_clauses[rng.integers(len(unsat_clauses))]

            # Decide: with prob p, pick a literal from this clause; else pick random variable
            if rng.random() < p:
                # Pick a literal from the unsatisfied clause
                lit_j = rng.integers(3)
                v = var_idx[uc, lit_j]
            else:
                v = rng.integers(n)

            # Flip variable v
            old_val = assignment[v]
            new_val = not old_val
            assignment[v] = new_val

            # Update sat_count for all clauses containing v
            for (ci, sig) in var_clauses[v]:
                # Before flip: was this literal satisfied?
                # old: satisfied if sig * (2*old_val - 1) > 0
                # new: satisfied if sig * (2*new_val - 1) > 0
                old_sat = 1 if sig * (1 if old_val else -1) > 0 else 0
                new_sat = 1 if sig * (1 if new_val else -1) > 0 else 0
                delta = new_sat - old_sat
                if delta != 0:
                    old_sc = sat_count[ci]
                    new_sc = old_sc + delta
                    sat_count[ci] = new_sc
                    # Track unsat set
                    if old_sc == 0 and new_sc > 0:
                        # Clause became satisfied
                        pass
                    elif old_sc > 0 and new_sc == 0:
                        # Clause became unsatisfied
                        pass

            # Recompute unsat_clauses (could be optimized but this is clear)
            unsat_clauses = np.where(sat_count == 0)[0]
            cur_unsat = len(unsat_clauses)

            # Adaptive p: increase randomness when stuck
            if step % 1000 == 999:
                p = 0.3 + 0.4 * rng.random()

        # End of this restart
        if best_assignment is not None and best_unsat == 0:
            return best_assignment.tolist()

    # Time's up: return best found
    if best_assignment is not None:
        return best_assignment.tolist()
    return [True] * n

# EVOLVE-BLOCK-END
