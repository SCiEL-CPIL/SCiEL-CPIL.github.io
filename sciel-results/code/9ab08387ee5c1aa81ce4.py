"""Rewritten 3-SAT solver: Min-Conflicts local search with deadline guard + polish."""
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    if m == 0:
        return [True] * n

    # ---- Precompute structures for fast evaluation/updating ----
    # clause_vars[i]: list of variable indices (0-based) in clause i
    # clause_sign[i]: list of booleans (True if literal positive)
    clause_vars = []
    clause_sign = []
    for c in clauses:
        vs = []
        ss = []
        for lit in c:
            vs.append(abs(lit) - 1)
            ss.append(lit > 0)
        clause_vars.append(vs)
        clause_sign.append(ss)

    # var_to_clauses[v]: list of clause indices containing variable v
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for v in clause_vars[i]:
            var_to_clauses[v].append(i)

    # ---- Initial assignment: random, biased toward satisfying more on average ----
    rng = random.Random(0)
    assignment = [rng.random() < 0.5 for _ in range(n)]

    # ---- Compute initial clause satisfaction and per-variable conflicts ----
    # clause_sat[i] = True if clause i is satisfied
    # var_conflicts[v] = number of unsatisfied clauses containing v
    clause_sat = [False] * m
    var_conflicts = [0] * n

    def compute_initial():
        for i in range(m):
            vs = clause_vars[i]
            ss = clause_sign[i]
            sat = False
            for k in range(len(vs)):
                if assignment[vs[k]] == ss[k]:
                    sat = True
                    break
            clause_sat[i] = sat
            if not sat:
                for v in vs:
                    var_conflicts[v] += 1

    compute_initial()

    def total_unsat():
        # count unsatisfied clauses
        cnt = 0
        for i in range(m):
            if not clause_sat[i]:
                cnt += 1
        return cnt

    def is_all_satisfied():
        for i in range(m):
            if not clause_sat[i]:
                return False
        return True

    def flip_var(v):
        # Flip assignment[v] and update clause_sat + var_conflicts incrementally.
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val
        for i in var_to_clauses[v]:
            vs = clause_vars[i]
            ss = clause_sign[i]
            was_sat = clause_sat[i]
            # Determine if this variable satisfied the clause before/after
            k = vs.index(v)  # small (<=3), acceptable
            lit_sat_before = (old_val == ss[k])
            lit_sat_after = (new_val == ss[k])
            if was_sat:
                if lit_sat_before and not lit_sat_after:
                    # clause may become unsat: check if any other literal now satisfies
                    still_sat = False
                    for j in range(len(vs)):
                        if j == k:
                            continue
                        if assignment[vs[j]] == ss[j]:
                            still_sat = True
                            break
                    if not still_sat:
                        clause_sat[i] = False
                        for vv in vs:
                            var_conflicts[vv] += 1
            else:
                if not lit_sat_before and lit_sat_after:
                    # clause becomes satisfied
                    clause_sat[i] = True
                    for vv in vs:
                        var_conflicts[vv] -= 1

    def min_conflicts_step():
        # Collect candidate variables from unsatisfied clauses
        cand = set()
        for i in range(m):
            if not clause_sat[i]:
                for v in clause_vars[i]:
                    cand.add(v)
        if not cand:
            return True
        # Pick variable minimizing resulting conflicts (evaluated via trial flip)
        best_v = None
        best_delta = None
        best_score = None
        # To keep it fast, sample a subset if candidate set is large
        cand_list = list(cand)
        if len(cand_list) > 64:
            cand_list = rng.sample(cand_list, 64)
        for v in cand_list:
            # Trial flip: compute delta in conflicts
            # delta = change in var_conflicts of v's clauses
            # We simulate: flipping v changes clause_sat for clauses in var_to_clauses[v]
            delta = 0
            for i in var_to_clauses[v]:
                vs = clause_vars[i]
                ss = clause_sign[i]
                k = vs.index(v)
                lit_sat_before = (assignment[v] == ss[k])
                lit_sat_after = (not assignment[v] == ss[k])
                was_sat = clause_sat[i]
                if was_sat:
                    if lit_sat_before and not lit_sat_after:
                        # would become unsat unless another lit satisfies
                        still_sat = False
                        for j in range(len(vs)):
                            if j == k:
                                continue
                            if assignment[vs[j]] == ss[j]:
                                still_sat = True
                                break
                        if not still_sat:
                            delta += 1
                else:
                    if not lit_sat_before and lit_sat_after:
                        delta -= 1
            # score = delta + small random noise for tie-breaking (soft surrogate)
            score = delta + rng.random() * 0.5
            if best_score is None or score < best_score:
                best_score = score
                best_v = v
                best_delta = delta
        if best_v is not None:
            flip_var(best_v)
        return is_all_satisfied()

    # ---- Deadline guard ----
    t_start = time.time()
    deadline = t_start + 295.0  # 5s safety margin before 300s
    best_assignment = list(assignment)
    best_unsat = total_unsat()

    # Main Min-Conflicts loop
    max_iter = 2000000
    it = 0
    while it < max_iter:
        if is_all_satisfied():
            break
        if time.time() > deadline:
            break
        min_conflicts_step()
        it += 1
        # Periodically update best
        if it % 500 == 0:
            u = total_unsat()
            if u < best_unsat:
                best_unsat = u
                best_assignment = list(assignment)
            if u == 0:
                break

    # Final check
    if not is_all_satisfied():
        u = total_unsat()
        if u < best_unsat:
            best_unsat = u
            best_assignment = list(assignment)

    # ---- Final high-precision polish pass if time remains and not solved ----
    if best_unsat > 0 and time.time() < deadline - 5.0:
        # Restore best assignment and do a tighter local search
        assignment = list(best_assignment)
        # Recompute clause_sat and var_conflicts for restored assignment
        for i in range(m):
            vs = clause_vars[i]
            ss = clause_sign[i]
            sat = False
            for k in range(len(vs)):
                if assignment[vs[k]] == ss[k]:
                    sat = True
                    break
            clause_sat[i] = sat
        for v in range(n):
            var_conflicts[v] = 0
        for i in range(m):
            if not clause_sat[i]:
                for v in clause_vars[i]:
                    var_conflicts[v] += 1

        polish_deadline = deadline - 2.0
        while time.time() < polish_deadline:
            if is_all_satisfied():
                break
            min_conflicts_step()
            if time.time() > polish_deadline:
                break
        u = total_unsat()
        if u < best_unsat:
            best_unsat = u
            best_assignment = list(assignment)

    return best_assignment
