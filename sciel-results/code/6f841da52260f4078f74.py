"""Synthetic planted 3-SAT solver (50 variables)."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290.0

    # Precompute clause info: for each clause, the variable indices and signs
    # clause_vars[i] = list of variable indices (0-based)
    # clause_sgn[i]  = list of +1/-1 (whether literal is positive)
    clause_vars = []
    clause_sgn = []
    for c in clauses:
        vs = []
        sgn = []
        for l in c:
            v = abs(l) - 1
            vs.append(v)
            sgn.append(1 if l > 0 else -1)
        clause_vars.append(vs)
        clause_sgn.append(sgn)

    def satisfied(assignment, i):
        vs = clause_vars[i]
        sgn = clause_sgn[i]
        for v, s in zip(vs, sgn):
            if (assignment[v] == 1) == (s == 1):
                return True
        return False

    def count_unsat(assignment):
        cnt = 0
        for i in range(m):
            if not satisfied(assignment, i):
                cnt += 1
        return cnt

    def pick_clause(assignment, unsat_indices):
        # choose an unsatisfied clause, weighted by fewest literals (all 3)
        return unsat_indices[random.randrange(len(unsat_indices))]

    def flip_var(assignment, v):
        assignment[v] = 1 - assignment[v]

    def run_walksat(rng, iters, alpha0, alpha_max, walk_ramp):
        # assignment as list of 0/1
        assignment = [rng.random() < 0.5 and 1 or 0 for _ in range(n)]
        alpha = alpha0
        best_unsat = m
        best_assignment = assignment[:]
        stall = 0
        for it in range(iters):
            if time.time() > deadline:
                break
            # find unsatisfied clauses
            unsat = []
            for i in range(m):
                if not satisfied(assignment, i):
                    unsat.append(i)
            if not unsat:
                return assignment
            if len(unsat) < best_unsat:
                best_unsat = len(unsat)
                best_assignment = assignment[:]
                stall = 0
            else:
                stall += 1

            # dynamic walk parameter
            alpha = min(alpha_max, alpha0 + (stall / max(1, walk_ramp)) * (alpha_max - alpha0))

            ci = random.choice(unsat)
            vs = clause_vars[ci]
            sgn = clause_sgn[ci]
            # find which literals are false (candidates to flip)
            candidates = []
            for v, s in zip(vs, sgn):
                if (assignment[v] == 1) != (s == 1):
                    candidates.append(v)
            if not candidates:
                candidates = vs
            if rng.random() < alpha:
                v = random.choice(candidates)
            else:
                # pick the variable whose flip improves the most
                best_v = candidates[0]
                best_gain = -1
                for v in candidates:
                    # count effect on other clauses: simple heuristic
                    gain = 0
                    for j in range(m):
                        if j == ci:
                            continue
                        vsj = clause_vars[j]
                        if v in vsj:
                            idx = vsj.index(v)
                            sj = clause_sgn[j][idx]
                            was_sat = (assignment[v] == 1) == (sj == 1)
                            now_sat = (1 - assignment[v] == 1) == (sj == 1)
                            # if was satisfied and becomes unsat, penalty
                            if was_sat and not now_sat:
                                gain -= 1
                            elif not was_sat and now_sat:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                v = best_v
            flip_var(assignment, v)

        return best_assignment

    # Restart policy
    rng = random.Random(0)
    best_assignment = None
    best_unsat = m

    # Try a few strategies with increasing walk aggressiveness
    for attempt in range(60):
        if time.time() > deadline:
            break
        if attempt < 10:
            res = run_walksat(rng, 3000, 0.2, 0.5, 200)
        elif attempt < 30:
            res = run_walksat(rng, 5000, 0.3, 0.7, 150)
        else:
            res = run_walksat(rng, 8000, 0.5, 0.9, 100)

        u = count_unsat(res)
        if u < best_unsat:
            best_unsat = u
            best_assignment = res
        if u == 0:
            break

    if best_assignment is None:
        best_assignment = [0] * n

    # Convert 0/1 to booleans
    return [bool(x) for x in best_assignment]
# EVOLVE-BLOCK-END
