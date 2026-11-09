"""Rewritten planted 3-SAT solver with vectorized WalkSAT."""
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

    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill

    # Build clause structure
    clause_vars = []
    clause_signs = []
    for c in clauses:
        clause_vars.append([abs(l) - 1 for l in c])
        clause_signs.append([1 if l > 0 else -1 for l in c])

    cv = np.array(clause_vars, dtype=np.int32)      # (m,3)
    cs = np.array(clause_signs, dtype=np.int32)      # (m,3)

    # For each variable: list of (clause_idx, sign) it appears in
    var_clauses = [[] for _ in range(n)]
    var_signs = [[] for _ in range(n)]
    for ci in range(m):
        for j in range(3):
            v = int(cv[ci, j])
            var_clauses[v].append(ci)
            var_signs[v].append(int(cs[ci, j]))

    # Precompute per-variable numpy arrays for fast vectorized cost
    vci = [np.array(var_clauses[v], dtype=np.int32) for v in range(n)]
    vsg = [np.array(var_signs[v], dtype=np.int32) for v in range(n)]

    # Per clause: for each position j, the other two variable indices and signs
    other = np.empty((m, 3, 2), dtype=np.int32)
    other_signs = np.empty((m, 3, 2), dtype=np.int32)
    for ci in range(m):
        for j in range(3):
            k = 0
            for t in range(3):
                if t != j:
                    other[ci, j, k] = cv[ci, t]
                    other_signs[ci, j, k] = cs[ci, t]
                    k += 1

    def is_satisfied(assignment):
        a = np.where(assignment, 1, -1)
        sat = a[cv] * cs
        return bool(np.all(np.max(sat, axis=1) == 1))

    def unsat_count(assignment):
        a = np.where(assignment, 1, -1)
        sat = a[cv] * cs
        return int(np.sum(np.max(sat, axis=1) != 1))

    def unsat_mask(assignment):
        a = np.where(assignment, 1, -1)
        sat = a[cv] * cs
        return np.max(sat, axis=1) != 1

    def walksat(assignment, max_steps, rng):
        p_flip = 0.5
        a = np.where(assignment, 1, -1).copy()
        unsat = unsat_mask(assignment)
        cur_unsat = int(np.sum(unsat))

        if cur_unsat == 0:
            assignment[:] = (a == 1)
            return assignment, True

        for step in range(max_steps):
            if cur_unsat == 0:
                assignment[:] = (a == 1)
                return assignment, True
            if (step % 200 == 0) and (time.time() > deadline):
                assignment[:] = (a == 1)
                return assignment, False

            # Pick a random unsatisfied clause
            while True:
                ci = rng.randrange(m)
                if unsat[ci]:
                    break

            if rng.random() < p_flip:
                j = rng.randrange(3)
                v = int(cv[ci, j])
            else:
                best_v = -1
                best_cost = 1 << 30
                for v in cv[ci].tolist():
                    cidx = vci[v]
                    if len(cidx) == 0:
                        continue
                    sgn = vsg[v]
                    val_v = a[v]
                    ovars = other[cidx]
                    ovals = a[ovars]
                    ovars_signs = other_signs[cidx]
                    affected_sat = ~unsat[cidx]
                    v_lit_true = (val_v * sgn) == 1
                    o_lit_true = (ovals * ovars_signs) == 1
                    other_all_false = np.all(~o_lit_true, axis=1)
                    critical = affected_sat & v_lit_true & other_all_false
                    cost = int(np.sum(critical))
                    if cost < best_cost:
                        best_cost = cost
                        best_v = v
                v = best_v if best_v >= 0 else int(cv[ci, 0])

            # Flip v
            a[v] = -a[v]

            # Update unsat for affected clauses
            cidx = vci[v]
            if len(cidx) > 0:
                sub_sat = np.max(a[cv[cidx]] * cs[cidx], axis=1) == 1
                old_unsat = unsat[cidx]
                new_unsat = ~sub_sat
                cur_unsat += int(np.sum(new_unsat)) - int(np.sum(old_unsat))
                unsat[cidx] = new_unsat

        assignment[:] = (a == 1)
        return assignment, False

    best_assignment = None
    best_unsat_count = m + 1
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        rng = random.Random(num_restarts * 12345 + 67890)
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        if is_satisfied(assignment):
            return assignment.tolist()
        max_steps = 200000
        assignment, solved = walksat(assignment, max_steps, rng)
        if solved:
            return assignment.tolist()
        uc = unsat_count(assignment)
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment.copy()

    if best_assignment is not None:
        return best_assignment.tolist()
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
