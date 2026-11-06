"""Combined solver: fast numpy WalkSAT with min-unsat heuristic and restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290.0  # safety margin before 300s kill

    # Precompute 0-based variable indices and signs as numpy arrays.
    # cvar[i] = (3,) int array of variable indices
    # csign[i] = (3,) int array of +1/-1
    cvar = np.empty((m, 3), dtype=np.int32)
    csign = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            cvar[i, j] = abs(lit) - 1
            csign[i, j] = 1 if lit > 0 else -1

    # Distinct variable indices per clause (for candidate selection).
    cvar_distinct = [list(dict.fromkeys(cvar[i])) for i in range(m)]

    # Reverse mapping: variable -> list of clause indices containing it.
    var_to_clauses = [[] for _ in range(n)]
    # Also map (clause, var) -> position for fast sign lookup.
    var_pos_in_clause = {}
    for i in range(m):
        for j in range(3):
            v = int(cvar[i, j])
            var_to_clauses[v].append(i)
            var_pos_in_clause[(i, v)] = j

    # Assignment as numpy bool array.
    rng = np.random.default_rng(0)

    def is_sat(a):
        # True if every clause has at least one true literal.
        lit_true = a[cvar] * (csign > 0)  # (m,3) bool: True where literal is true
        return not np.any(~lit_true.any(axis=1))

    def init_sat_count(a):
        # sat_count[i] = number of true literals in clause i.
        lit_true = a[cvar] * (csign > 0)
        return lit_true.sum(axis=1)

    def unsat_indices(sc):
        return np.nonzero(sc == 0)[0]

    # Initial random assignment.
    a = rng.random(n) < 0.5
    sc = init_sat_count(a)
    if not np.any(sc == 0):
        return a.tolist()

    # Local search with min-unsat heuristic + randomization + restarts.
    # We track sat_count incrementally.
    total_flips = 0
    restart_count = 0
    max_steps_per_run = 3000
    steps_in_run = 0

    while time.time() < deadline:
        unsat = unsat_indices(sc)
        if len(unsat) == 0:
            break

        # Pick a random unsatisfied clause.
        ci = int(unsat[rng.integers(len(unsat))])
        cand_vars = cvar_distinct[ci]
        if not cand_vars:
            break

        # Compute delta (change in #unsat clauses) for flipping each candidate.
        # delta = (#clauses that become unsat) - (#clauses that become sat)
        best_var = cand_vars[0]
        best_delta = None
        for v in cand_vars:
            cur = a[v]
            delta = 0
            for cj in var_to_clauses[v]:
                pos = var_pos_in_clause[(cj, v)]
                sgn = csign[cj, pos]
                # Is v's literal currently satisfied in clause cj?
                sat_lit = (cur == (sgn > 0))
                if sat_lit:
                    # Flipping makes it false: sat_count decreases.
                    # If it was the only true literal, clause becomes unsat.
                    if sc[cj] == 1:
                        delta += 1
                else:
                    # Flipping makes it true: sat_count increases.
                    # If clause was fully false, it becomes sat.
                    if sc[cj] == 0:
                        delta -= 1
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_var = v

        # Randomization: with 30% probability pick a random candidate.
        if rng.random() < 0.3:
            v = cand_vars[rng.integers(len(cand_vars))]
        else:
            v = best_var

        # Apply flip.
        a[v] = not a[v]
        for cj in var_to_clauses[v]:
            pos = var_pos_in_clause[(cj, v)]
            sgn = csign[cj, pos]
            if a[v] == (sgn > 0):
                sc[cj] += 1
            else:
                sc[cj] -= 1

        total_flips += 1
        steps_in_run += 1

        # Restart policy: geometric restarts.
        if steps_in_run >= max_steps_per_run:
            unsat_now = unsat_indices(sc)
            if len(unsat_now) > 0:
                restart_count += 1
                # Increase max_steps geometrically, capped.
                max_steps_per_run = min(int(max_steps_per_run * 1.5), 50000)
                steps_in_run = 0
                # Occasionally re-randomize the whole assignment.
                if restart_count % 3 == 0:
                    a = rng.random(n) < 0.5
                    sc = init_sat_count(a)
                    if not np.any(sc == 0):
                        break

    return a.tolist()
# EVOLVE-BLOCK-END
