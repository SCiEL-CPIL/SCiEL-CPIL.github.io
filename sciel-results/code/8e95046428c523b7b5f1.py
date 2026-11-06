"""
Planted 3-SAT solver using a robust WalkSAT with structural initialization,
greedy refinement, and a final high-precision polish pass.
"""
import time
import random

DEADLINE_FRACTION = 0.95

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    t_start = time.time()
    deadline = t_start + 330 * DEADLINE_FRACTION
    
    # Preprocess: Convert to 0-based internal representation
    clause_vars = []
    for clause in clauses:
        cv = []
        for l in clause:
            if l > 0:
                cv.append((l - 1, True))
            else:
                cv.append((-l - 1, False))
        clause_vars.append(cv)

    var_clauses = [[] for _ in range(n)]
    for ci, cvars in enumerate(clause_vars):
        for v, sign in cvars:
            var_clauses[v].append((ci, sign))

    # Structural bias
    pos_count = [0] * n
    neg_count = [0] * n
    for cvars in clause_vars:
        for v, sign in cvars:
            if sign:
                pos_count[v] += 1
            else:
                neg_count[v] += 1

    rng = random.Random(42)

    def initial_assignment():
        assignment = []
        for v in range(n):
            if pos_count[v] > neg_count[v]:
                p = 0.6
            elif neg_count[v] > pos_count[v]:
                p = 0.4
            else:
                p = 0.5
            assignment.append(rng.random() < p)
        return assignment

    def make_support_counts(assignment):
        support = [0] * m
        for ci, cvars in enumerate(clause_vars):
            s = 0
            for v, sign in cvars:
                if assignment[v] == sign:
                    s += 1
            support[ci] = s
        return support

    def apply_flip(v, assignment, support, cur_unsat):
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val
        delta_unsat = 0
        for ci, sign in var_clauses[v]:
            old_support = support[ci]
            if new_val == sign:
                new_support = old_support + 1
            else:
                new_support = old_support - 1
            support[ci] = new_support
            if old_support == 0 and new_support == 1:
                delta_unsat -= 1
            elif old_support == 1 and new_support == 0:
                delta_unsat += 1
        return cur_unsat + delta_unsat

    def greedy_polish(assignment, support, cur_unsat):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            if cur_unsat == 0:
                return assignment, support, cur_unsat
            best_gain = 0
            best_v = -1
            for v in range(n):
                gain = 0
                for ci, sign in var_clauses[v]:
                    if assignment[v] == sign:
                        if support[ci] == 1:
                            gain -= 1
                    else:
                        if support[ci] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_gain > 0:
                cur_unsat = apply_flip(best_v, assignment, support, cur_unsat)
                improved = True
                if cur_unsat == 0:
                    break
        return assignment, support, cur_unsat

    best_assignment = None
    best_unsat = m + 1

    # WalkSAT Parameters
    noise = 0.5
    max_steps_per_restart = 50000
    restarts = 200

    for restart in range(restarts):
        if time.time() > deadline:
            break

        current_noise = 0.5 + 0.1 * (restart % 3 - 1)
        if current_noise < 0.1: current_noise = 0.1
        if current_noise > 0.9: current_noise = 0.9

        assignment = initial_assignment()
        support = make_support_counts(assignment)
        cur_unsat = sum(1 for s in support if s == 0)

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break

            if cur_unsat == 0:
                return assignment

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            # Pick random unsatisfied clause
            while True:
                ci = rng.randint(0, m - 1)
                if support[ci] == 0:
                    break

            # Determine which variable to flip in this clause
            if rng.random() < current_noise:
                idx = rng.randint(0, 2)
                v, sign = clause_vars[ci][idx]
            else:
                best_gain = -999
                best_v = -1
                for idx in range(3):
                    v, sign = clause_vars[ci][idx]
                    gain = 0
                    for ci2, sign2 in var_clauses[v]:
                        if ci2 == ci:
                            continue
                        if not assignment[v] == sign2:
                            if support[ci2] == 0:
                                gain += 1
                        else:
                            if support[ci2] == 1:
                                gain -= 1
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v

                v, sign = best_v, clause_vars[ci][0][1] # dummy sign, only v matters

            cur_unsat = apply_flip(v, assignment, support, cur_unsat)

            if cur_unsat == 0:
                return assignment

    # Final Polish: Local Search (Greedy)
    if best_assignment is not None and time.time() < deadline:
        assignment = best_assignment[:]
        support = make_support_counts(assignment)
        cur_unsat = sum(1 for s in support if s == 0)

        assignment, support, cur_unsat = greedy_polish(assignment, support, cur_unsat)

        if cur_unsat == 0:
            return assignment

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
