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
    m = len(clauses)

    start = time.time()
    deadline = start + 295.0  # safety margin before the 330s kill

    rng = random.Random(0)

    # Precompute literal arrays.
    lit_pos = np.empty((m, 3), dtype=np.int64)   # variable index (0-based)
    lit_sign = np.empty((m, 3), dtype=np.int8)   # +1 / -1
    for i, cl in enumerate(clauses):
        for j, l in enumerate(cl):
            lit_pos[i, j] = abs(l) - 1
            lit_sign[i, j] = 1 if l > 0 else -1

    # Map each variable -> list of (clause_index, literal_slot) it appears in.
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(lit_pos[i, j])
            var_to_clauses[v].append((i, j))

    def build_value(assignment):
        return np.where(np.asarray(assignment, dtype=bool), 1, -1).astype(np.int8)

    def clause_sat_mask(value):
        return np.any(lit_sign * value[lit_pos] == 1, axis=1)

    # Initial random assignment.
    assignment = [rng.random() < 0.5 for _ in range(n)]
    value = build_value(assignment)
    unsat_mask = ~clause_sat_mask(value)
    cur_unsat = int(np.count_nonzero(unsat_mask))
    if cur_unsat == 0:
        return assignment

    best_unsat = cur_unsat
    best_assignment = assignment[:]

    noise = 0.35
    max_steps_per_restart = 40000

    while time.time() < deadline:
        # Random restart.
        assignment = [rng.random() < 0.5 for _ in range(n)]
        value = build_value(assignment)
        unsat_mask = ~clause_sat_mask(value)
        cur_unsat = int(np.count_nonzero(unsat_mask))
        if cur_unsat == 0:
            return assignment
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        # Occasionally seed from the best-so-far when it's very close.
        if best_unsat <= 2 and rng.random() < 0.3:
            assignment = best_assignment[:]
            value = build_value(assignment)
            unsat_mask = ~clause_sat_mask(value)
            cur_unsat = best_unsat

        for _ in range(max_steps_per_restart):
            if time.time() >= deadline:
                break
            if cur_unsat == 0:
                return assignment

            unsat_indices = np.nonzero(unsat_mask)[0]
            ci = rng.choice(unsat_indices.tolist())

            # Compute net gain for each of the 3 literals in clause ci:
            # flipping var v changes satisfaction of every clause containing v.
            # gain = (# unsat clauses containing v that become sat)
            #       - (# sat clauses containing v that become unsat)
            gains = np.zeros(3, dtype=np.int64)
            for j in range(3):
                v = int(lit_pos[ci, j])
                s = int(lit_sign[ci, j])
                cur_val = value[v]
                # After flip, literal (v, s) value becomes -s*cur_val.
                # For each clause k containing v at slot t with sign st:
                #   current lit value = st * cur_val
                #   new lit value     = st * (-cur_val)
                # A clause is satisfied iff any of its 3 literals == 1.
                fixed = 0
                broken = 0
                for (k, t) in var_to_clauses[v]:
                    st = int(lit_sign[k, t])
                    cur_lit = st * cur_val
                    new_lit = -cur_lit
                    # Determine sat status before and after for clause k.
                    # Before:
                    sat_before = (cur_lit == 1)
                    for t2 in range(3):
                        if t2 == t:
                            continue
                        if int(lit_sign[k, t2]) * value[int(lit_pos[k, t2])] == 1:
                            sat_before = True
                            break
                    # After:
                    sat_after = (new_lit == 1)
                    for t2 in range(3):
                        if t2 == t:
                            continue
                        if int(lit_sign[k, t2]) * value[int(lit_pos[k, t2])] == 1:
                            sat_after = True
                            break
                    if not sat_before and sat_after:
                        fixed += 1
                    elif sat_before and not sat_after:
                        broken += 1
                gains[j] = fixed - broken

            if rng.random() < noise:
                j = rng.randrange(3)
            else:
                maxg = int(gains.max())
                candidates = np.nonzero(gains == maxg)[0]
                j = rng.choice(candidates.tolist())

            v = int(lit_pos[ci, j])
            value[v] = -value[v]
            assignment[v] = value[v] > 0

            # Incrementally update unsat_mask for clauses containing v.
            for (k, t) in var_to_clauses[v]:
                sat = np.any(lit_sign[k] * value[lit_pos[k]] == 1)
                if sat:
                    if unsat_mask[k]:
                        cur_unsat -= 1
                        unsat_mask[k] = False
                else:
                    if not unsat_mask[k]:
                        cur_unsat += 1
                        unsat_mask[k] = True

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

    return best_assignment
# EVOLVE-BLOCK-END
