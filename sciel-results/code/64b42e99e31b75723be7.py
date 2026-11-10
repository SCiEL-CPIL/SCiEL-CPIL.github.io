"""Planted 3-SAT solver: planted-recovery + min-conflicts + multi-restart."""
# EVOLVE-BLOCK-START
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    deadline = time.time() + 285.0  # safety margin vs 300s evaluator / 330s kill

    # ---- Build occurrence maps (signed literal -> list of clause indices) ----
    lit2cl = {}
    for ci, cl in enumerate(clauses):
        for lit in cl:
            lit2cl.setdefault(lit, []).append(ci)

    def check(assignment):
        for cl in clauses:
            ok = False
            for lit in cl:
                v = assignment[abs(lit) - 1]
                if (lit > 0) == v:
                    ok = True
                    break
            if not ok:
                return False
        return True

    # ---- Attempt 1: planted-solution recovery (deterministic) ----
    # Heuristic: for a variable, the literal that appears in MORE clauses is
    # more likely to be the planted true literal. Resolve conflicts by flipping
    # the least-supported literal.
    try:
        occ_pos = [0] * (n + 1)
        occ_neg = [0] * (n + 1)
        for lit, lst in lit2cl.items():
            if lit > 0:
                occ_pos[lit] += len(lst)
            else:
                occ_neg[-lit] += len(lst)
        planted = [False] * n
        for i in range(1, n + 1):
            planted[i - 1] = occ_pos[i] >= occ_neg[i]
        # quick local repair: flip the least-supported literal in unsat clauses
        for _ in range(400):
            unsat = [cl for cl in clauses
                     if not any((lit > 0) == planted[abs(lit) - 1] for lit in cl)]
            if not unsat:
                break
            # pick clause, flip its least-supported literal
            cl = unsat[0]
            best_lit = None
            best_score = None
            for lit in cl:
                var = abs(lit)
                score = occ_pos[var] if lit > 0 else occ_neg[var]
                if best_score is None or score < best_score:
                    best_score = score
                    best_lit = lit
            planted[abs(best_lit) - 1] = not planted[abs(best_lit) - 1]
        if check(planted):
            return planted
    except Exception:
        pass

    # ---- Min-conflicts local search with weighted literal selection ----
    # Precompute per-clause variable lists and sign flags for speed.
    cl_vars = [[abs(lit) - 1 for lit in cl] for cl in clauses]
    cl_sign = [[(lit > 0) for lit in cl] for cl in clauses]
    # var -> list of (clause_idx, sign)
    var_occ = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            var_occ[abs(lit) - 1].append((ci, lit > 0))

    def min_conflicts(assignment, rng):
        # compute per-clause satisfied flag
        sat = [False] * m
        for ci in range(m):
            cv = cl_vars[ci]
            cs = cl_sign[ci]
            for k in range(3):
                if assignment[cv[k]] == cs[k]:
                    sat[ci] = True
                    break
        # per-variable conflict count (number of unsat clauses containing var)
        var_conf = [0] * n
        for ci in range(m):
            if not sat[ci]:
                for v in cl_vars[ci]:
                    var_conf[v] += 1
        for _ in range(30000):
            # find unsat clauses
            unsat_idx = [ci for ci in range(m) if not sat[ci]]
            if not unsat_idx:
                return assignment
            ci = unsat_idx[rng.randrange(len(unsat_idx))]
            cv = cl_vars[ci]
            cs = cl_sign[ci]
            # weight each literal by its variable's conflict count; prefer flipping
            # a variable whose flip would help (currently wrong in this clause)
            best_v = None
            best_w = -1.0
            for k in range(3):
                v = cv[k]
                currently_true = (assignment[v] == cs[k])
                w = var_conf[v] + (0.0 if currently_true else 1000.0)
                if w > best_w:
                    best_w = w
                    best_v = v
            # flip best_v
            old = assignment[best_v]
            new = not old
            assignment[best_v] = new
            # update sat and var_conf for all clauses containing best_v
            for (ci2, sign) in var_occ[best_v]:
                # recompute satisfaction of clause ci2
                cv2 = cl_vars[ci2]
                cs2 = cl_sign[ci2]
                sat2 = False
                for k in range(3):
                    if assignment[cv2[k]] == cs2[k]:
                        sat2 = True
                        break
                was_sat = sat[ci2]
                if sat2 != was_sat:
                    sat[ci2] = sat2
                    for v2 in cv2:
                        if sat2:
                            var_conf[v2] -= 1
                        else:
                            var_conf[v2] += 1
        return assignment

    import random
    rng = random.Random(0)

    # Multi-restart
    while time.time() < deadline:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        assignment = min_conflicts(assignment, rng)
        if check(assignment):
            return assignment
        # occasional planted-seeded restart
        if rng.random() < 0.3:
            try:
                seed = [occ_pos[i] >= occ_neg[i] for i in range(1, n + 1)]
                assignment = min_conflicts(seed, rng)
                if check(assignment):
                    return assignment
            except Exception:
                pass

    # Fallback: return last assignment (may be unsat)
    return assignment
# EVOLVE-BLOCK-END
