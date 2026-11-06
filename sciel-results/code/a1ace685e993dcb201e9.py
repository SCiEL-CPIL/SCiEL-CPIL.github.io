# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Precompute, for each clause, the literal indices into the assignment array.
    # idx[i] is the 0-based variable index; sign[i] is +1/-1.
    cidx = [[abs(l) - 1 for l in c] for c in clauses]
    csign = [[1 if l > 0 else -1 for l in c] for c in clauses]
    cvar = [list(set(idx)) for idx in cidx]  # distinct variables per clause

    # Build reverse mapping: variable -> list of clause indices containing it.
    var_to_clauses = [[] for _ in range(n)]
    for ci, idxs in enumerate(cidx):
        for v in cvar[ci]:
            var_to_clauses[v].append(ci)

    deadline = time.time() + 290.0  # safety margin before 330s kill
    rng = random.Random(0)

    def is_sat(assignment):
        for ci in range(m):
            ok = False
            idxs = cidx[ci]
            signs = csign[ci]
            for j in range(len(idxs)):
                if assignment[idxs[j]] == (signs[j] > 0):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def count_unsat(assignment):
        cnt = 0
        for ci in range(m):
            idxs = cidx[ci]
            signs = csign[ci]
            ok = False
            for j in range(len(idxs)):
                if assignment[idxs[j]] == (signs[j] > 0):
                    ok = True
                    break
            if not ok:
                cnt += 1
        return cnt

    # Random initial assignment.
    assignment = [rng.random() < 0.5 for _ in range(n)]
    if is_sat(assignment):
        return assignment

    # Local search: min-unsat with randomization and restarts.
    # unsat_count[ci] = number of falsified literals in clause ci (0..len).
    # A clause is satisfied iff unsat_count[ci] == len(cidx[ci]).
    # Actually track "satisfied count" per clause: sat_count[ci] = # true literals.
    sat_count = [0] * m
    for ci in range(m):
        idxs = cidx[ci]
        signs = csign[ci]
        for j in range(len(idxs)):
            if assignment[idxs[j]] == (signs[j] > 0):
                sat_count[ci] += 1

    def unsat_clauses():
        res = []
        for ci in range(m):
            if sat_count[ci] == 0:
                res.append(ci)
        return res

    max_steps_per_run = 5000
    total_flips = 0

    while time.time() < deadline:
        unsat = unsat_clauses()
        if not unsat:
            break
        # pick a random unsat clause
        ci = rng.choice(unsat)
        idxs = cidx[ci]
        signs = csign[ci]
        # candidate variables in this clause
        cand_vars = []
        for v in cvar[ci]:
            cand_vars.append(v)
        if not cand_vars:
            break
        # For each candidate variable, compute delta in #unsat clauses if flipped.
        best_var = None
        best_delta = None
        for v in cand_vars:
            cur = assignment[v]
            # flipping v: for each clause containing v, recompute contribution change.
            delta = 0
            for cj in var_to_clauses[v]:
                before = sat_count[cj]
                # determine v's sign in clause cj
                # find sign of v within cj
                sgn = None
                for j in range(len(cidx[cj])):
                    if cidx[cj][j] == v:
                        sgn = csign[cj][j]
                        break
                # literal satisfied currently?
                sat_lit = (cur == (sgn > 0))
                if sat_lit:
                    # flipping makes it false: sat_count decreases by 1
                    if before == 1:
                        # clause becomes unsat: +1 unsat
                        delta += 1
                else:
                    # flipping makes it true: sat_count increases by 1
                    if before == 0:
                        # clause becomes sat: -1 unsat
                        delta -= 1
                # update sat_count tentatively not needed; we recompute per var
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_var = v
        # randomization: with some probability pick a random candidate instead
        if rng.random() < 0.3:
            v = rng.choice(cand_vars)
        else:
            v = best_var
        # apply flip
        assignment[v] = not assignment[v]
        for cj in var_to_clauses[v]:
            sgn = None
            for j in range(len(cidx[cj])):
                if cidx[cj][j] == v:
                    sgn = csign[cj][j]
                    break
            if assignment[v] == (sgn > 0):
                sat_count[cj] += 1
            else:
                sat_count[cj] -= 1
        total_flips += 1

        if total_flips % 5000 == 0 and unsat_clauses():
            # restart with new random assignment occasionally
            if total_flips % 50000 == 0:
                assignment = [rng.random() < 0.5 for _ in range(n)]
                sat_count = [0] * m
                for ci in range(m):
                    idxs = cidx[ci]
                    signs = csign[ci]
                    for j in range(len(idxs)):
                        if assignment[idxs[j]] == (signs[j] > 0):
                            sat_count[ci] += 1
                break  # break inner run to re-evaluate

    return assignment
# EVOLVE-BLOCK-END
