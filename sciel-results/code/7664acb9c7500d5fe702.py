"""WalkSAT-family solver: weighted clause selection + min-conflicts scoring + SA escapes + restarts."""
# EVOLVE-BLOCK-START
import time
import math
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    deadline = time.time() + 290.0  # safety margin vs 330s kill

    # Precompute clause frequency (weight) and literal -> clause index list
    freq = [1] * m
    for c in clauses:
        for l in c:
            freq[abs(l) - 1] += 1  # placeholder, real below

    # real frequency: number of clauses each variable appears in
    var_freq = [0] * n
    for c in clauses:
        for l in c:
            var_freq[abs(l) - 1] += 1

    # For each variable, list of (clause_index, sign) where sign = +1 if literal positive
    var_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            var_clauses[v].append((ci, 1 if l > 0 else -1))

    # Precompute clause literal arrays and per-variable sign within clause for fast eval
    clause_vars = []   # list of (var_index, sign)
    for c in clauses:
        clause_vars.append([(abs(l) - 1, 1 if l > 0 else -1) for l in c])

    # Clause weight (soft frequency, capped) for selection
    alpha = 1.0
    w = [float(var_freq[clause_vars[i][0][0]] if clause_vars[i] else 1) ** alpha for i in range(m)]
    # normalize-ish
    wsum = sum(w)

    def clause_satisfied(ci, assignment):
        for v, s in clause_vars[ci]:
            if assignment[v] == (s > 0):
                return True
        return False

    def count_unsat(assignment):
        u = 0
        for ci in range(m):
            sat = False
            for v, s in clause_vars[ci]:
                if assignment[v] == (s > 0):
                    sat = True
                    break
            if not sat:
                u += 1
        return u

    def evaluate(assignment):
        # objective = sum over unsat clauses of weight (soft)
        obj = 0.0
        for ci in range(m):
            sat = False
            for v, s in clause_vars[ci]:
                if assignment[v] == (s > 0):
                    sat = True
                    break
            if not sat:
                obj += w[ci]
        return obj

    def best_flip_score(ci, assignment):
        # For each variable in clause ci, score = number of currently-satisfied clauses it would break
        # weighted by clause weight. Lower is better.
        best = None
        best_score = None
        for v, s in clause_vars[ci]:
            # if we flip v, which currently-satisfied clauses containing v become unsat?
            break_score = 0.0
            for (cj, sj) in var_clauses[v]:
                # clause cj currently satisfied?
                sat = False
                for (vv, ss) in clause_vars[cj]:
                    if assignment[vv] == (ss > 0):
                        sat = True
                        break
                if sat:
                    # after flipping v, does clause cj stay satisfied via another literal?
                    stays = False
                    for (vv, ss) in clause_vars[cj]:
                        if vv == v:
                            continue
                        if assignment[vv] == (ss > 0):
                            stays = True
                            break
                    if not stays:
                        break_score += w[cj]
            if best_score is None or break_score < best_score:
                best_score = break_score
                best = v
        return best, best_score

    best_obj = float('inf')
    best_assignment = None

    rng = random.Random(12345)
    step_budget = 60000
    T0 = 1.5
    T1 = 0.02

    while time.time() < deadline:
        # random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        obj = evaluate(assignment)
        if obj < best_obj:
            best_obj = obj
            best_assignment = assignment[:]
        if obj == 0.0:
            return assignment

        # SA schedule over this run
        steps = step_budget
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break
            # find unsat clauses
            unsat = []
            for ci in range(m):
                sat = False
                for v, s in clause_vars[ci]:
                    if assignment[v] == (s > 0):
                        sat = True
                        break
                if not sat:
                    unsat.append(ci)
            if not unsat:
                return assignment
            if len(unsat) == 0:
                return assignment

            # weighted clause selection
            total_w = sum(w[ci] for ci in unsat)
            r = rng.random() * total_w
            acc = 0.0
            chosen = unsat[-1]
            for ci in unsat:
                acc += w[ci]
                if r <= acc:
                    chosen = ci
                    break

            T = T0 * (T1 / T0) ** (step / max(1, steps - 1))

            # candidate flips: min-conflicts best + random
            v_best, _ = best_flip_score(chosen, assignment)
            # with some probability pick a random var in clause (diversity)
            if rng.random() < 0.25:
                v, s = clause_vars[chosen][rng.randrange(len(clause_vars[chosen]))]
            else:
                v = v_best

            old_obj = obj
            assignment[v] = not assignment[v]
            new_obj = evaluate(assignment)
            delta = new_obj - old_obj
            if delta <= 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                obj = new_obj
                if obj < best_obj:
                    best_obj = obj
                    best_assignment = assignment[:]
            else:
                assignment[v] = not assignment[v]

        # after run, if improved, keep going; else restart
        if best_obj == 0.0:
            return best_assignment

    return best_assignment if best_assignment is not None else [False] * n
# EVOLVE-BLOCK-END
