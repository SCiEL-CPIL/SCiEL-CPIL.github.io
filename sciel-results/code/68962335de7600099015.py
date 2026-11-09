"""Improved planted 3-SAT solver with efficient WalkSAT + restarts + DLS."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: 330s wall-clock limit, use 295s for safety
    start_time = time.time()
    deadline = start_time + 295.0

    # Preprocess: convert to 0-based variable indices and sign (+1/-1)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cv.append((v, s))
        clause_vars.append(cv)

    # Build variable -> list of (clause_idx, sign) for fast updates
    var_occurrences = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for pi, (v, s) in enumerate(cv):
            var_occurrences[v].append((ci, s))

    # Clause -> variables as flat arrays for faster access
    clause_var_ids = []
    clause_signs = []
    for cv in clause_vars:
        clause_var_ids.append([x[0] for x in cv])
        clause_signs.append([x[1] for x in cv])

    rng = random.Random(42)

    def full_check(assignment):
        """Check if all clauses are satisfied."""
        for ci in range(m):
            vlist = clause_var_ids[ci]
            slist = clause_signs[ci]
            sat = False
            for i in range(len(vlist)):
                v = vlist[i]
                s = slist[i]
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    sat = True
                    break
            if not sat:
                return False, ci
        return True, -1

    def init_clause_sat_count(assignment):
        """Initialize clause_sat_count."""
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for ci in range(m):
            vlist = clause_var_ids[ci]
            slist = clause_signs[ci]
            cnt = 0
            for i in range(len(vlist)):
                v = vlist[i]
                s = slist[i]
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    cnt += 1
            clause_sat_count[ci] = cnt
        return clause_sat_count

    def pick_unsat_clause(clause_sat_count, rng):
        """Pick a random unsatisfied clause efficiently."""
        unsat_indices = np.where(clause_sat_count == 0)[0]
        if len(unsat_indices) == 0:
            return -1
        return int(unsat_indices[rng.randint(0, len(unsat_indices) - 1)])

    def compute_delta_unsat(v, assignment, clause_sat_count):
        """Compute change in unsat count if we flip variable v."""
        delta = 0
        v_val = assignment[v]
        for (aci, s) in var_occurrences[v]:
            lit_true = (v_val and s == 1) or ((not v_val) and s == -1)
            if lit_true:
                if clause_sat_count[aci] == 1:
                    delta += 1
            else:
                if clause_sat_count[aci] == 0:
                    delta -= 1
        return delta

    def do_flip(v, assignment, clause_sat_count):
        """Flip variable v and update clause_sat_count."""
        old_val = assignment[v]
        assignment[v] = not old_val
        for (aci, s) in var_occurrences[v]:
            was_sat = (old_val and s == 1) or ((not old_val) and s == -1)
            is_sat = (assignment[v] and s == 1) or ((not assignment[v] and s == -1))
            if was_sat and not is_sat:
                clause_sat_count[aci] -= 1
            elif (not was_sat) and is_sat:
                clause_sat_count[aci] += 1

    def polish(assignment, clause_sat_count, time_limit):
        """Final polish pass: try to satisfy remaining clauses via targeted flips."""
        unsat_count = int(np.sum(clause_sat_count == 0))
        if unsat_count == 0:
            return assignment, clause_sat_count, 0
        t_start = time.time()
        for _ in range(50000):
            if time.time() > time_limit:
                break
            if unsat_count == 0:
                break
            unsat_idx = np.where(clause_sat_count == 0)[0]
            if len(unsat_idx) == 0:
                break
            ci = int(unsat_idx[rng.randint(0, len(unsat_idx) - 1)])
            vlist = clause_var_ids[ci]
            slist = clause_signs[ci]
            best_v = -1
            best_delta = 999
            for v in vlist:
                d = compute_delta_unsat(v, assignment, clause_sat_count)
                if d < best_delta:
                    best_delta = d
                    best_v = v
            if best_v >= 0 and best_delta <= 0:
                do_flip(best_v, assignment, clause_sat_count)
                unsat_count += best_delta
            elif best_v >= 0 and rng.random() < 0.3:
                do_flip(best_v, assignment, clause_sat_count)
                unsat_count += best_delta
            else:
                break
        return assignment, clause_sat_count, unsat_count

    # Variable activity for DLS-like heuristic
    var_activity = np.zeros(n, dtype=np.float64)
    activity_decay = 0.995

    best_assignment = None
    best_unsat_count = m + 1

    base_steps = 3000
    max_restarts = 10000
    p_good = 0.6

    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break

        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        clause_sat_count = init_clause_sat_count(assignment)
        unsat_count = int(np.sum(clause_sat_count == 0))

        if unsat_count == 0:
            return assignment.tolist()

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()

        if unsat_count <= 5:
            steps = base_steps * 3
        elif unsat_count <= 20:
            steps = base_steps * 2
        else:
            steps = base_steps

        stagnant = 0
        for step in range(steps):
            if time.time() > deadline - 1.0:
                break

            if unsat_count == 0:
                return assignment.tolist()

            ci = pick_unsat_clause(clause_sat_count, rng)
            if ci < 0:
                return assignment.tolist()

            vlist = clause_var_ids[ci]
            slist = clause_signs[ci]
            clen = len(vlist)

            if rng.random() < p_good:
                candidates = []
                for i in range(clen):
                    v = vlist[i]
                    s = slist[i]
                    is_lit_sat = (assignment[v] and s == 1) or (not assignment[v] and s == -1)
                    if not is_lit_sat:
                        candidates.append(v)
                if candidates:
                    if len(candidates) > 1:
                        weights = [var_activity[v] + 1.0 for v in candidates]
                        total_w = sum(weights)
                        r = rng.random() * total_w
                        cum = 0
                        v = candidates[-1]
                        for j, cv in enumerate(candidates):
                            cum += weights[j]
                            if r <= cum:
                                v = cv
                                break
                    else:
                        v = candidates[0]
                else:
                    v = vlist[rng.randint(0, clen - 1)]
            else:
                if clen > 1:
                    weights = [var_activity[vlist[i]] + 1.0 for i in range(clen)]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    v = vlist[-1]
                    for i in range(clen):
                        cum += weights[i]
                        if r <= cum:
                            v = vlist[i]
                            break
                else:
                    v = vlist[0]

            delta = compute_delta_unsat(v, assignment, clause_sat_count)

            if delta <= 0 or (delta > 0 and rng.random() < 0.25):
                do_flip(v, assignment, clause_sat_count)
                new_unsat = unsat_count + delta
                if new_unsat < unsat_count:
                    stagnant = 0
                else:
                    stagnant += 1
                unsat_count = new_unsat

                for i in range(clen):
                    var_activity[vlist[i]] *= activity_decay
                    var_activity[vlist[i]] += 1.0

                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment.copy()
            else:
                stagnant += 1

            if stagnant > 200 and unsat_count > 10:
                break

        var_activity *= 0.95

        if time.time() > deadline - 1.0:
            break

    if best_assignment is not None and time.time() < deadline - 5.0:
        best_clause_sat = init_clause_sat_count(best_assignment)
        best_unsat = int(np.sum(best_clause_sat == 0))
        if best_unsat > 0 and best_unsat <= 20:
            polished, _, _ = polish(best_assignment, best_clause_sat, deadline - 2.0)
            final_unsat = int(np.sum(best_clause_sat == 0))
            if final_unsat == 0:
                return best_assignment.tolist()
            if best_unsat < unsat_count:
                best_assignment = polished

    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        return [False] * n
# EVOLVE-BLOCK-END
