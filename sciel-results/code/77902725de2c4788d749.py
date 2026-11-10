# EVOLVE-BLOCK-START
import time
import math
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285.0  # safety margin before 300s kill
    rng = random.Random(0)

    # Precompute per-variable clause incidence (for fast delta evaluation)
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        seen = set()
        for l in clause:
            v = abs(l) - 1
            if v not in seen:
                var_clauses[v].append(ci)
                seen.add(v)

    # Initial random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]

    # Clause status and per-literal satisfaction counts
    clause_satisfied = [False] * m
    lit_count = [[0, 0] for _ in range(m)]  # [pos_count, neg_count]

    for ci, clause in enumerate(clauses):
        pos = 0
        neg = 0
        for l in clause:
            if l > 0:
                if assignment[abs(l) - 1]:
                    pos += 1
            else:
                if not assignment[abs(l) - 1]:
                    neg += 1
        lit_count[ci][0] = pos
        lit_count[ci][1] = neg
        clause_satisfied[ci] = (pos + neg) > 0

    total_unsat = sum(1 for s in clause_satisfied if not s)

    best_unsat = total_unsat
    best_assignment = assignment[:]
    best_time = time.time()

    # Simulated annealing parameters
    T0 = 3.0
    Tend = 0.02
    # We'll step through a large number of iterations; temperature decays
    # geometrically. We pick total steps to roughly fill the budget.
    total_steps = 4000000
    if total_steps > 1:
        cooling = (Tend / T0) ** (1.0 / total_steps)
    else:
        cooling = 1.0

    T = T0
    step = 0
    no_improve = 0

    while time.time() < deadline:
        if step >= total_steps:
            break

        if total_unsat == 0:
            best_unsat = 0
            best_assignment = assignment[:]
            break

        # Choose a variable to flip: preferentially from unsatisfied clauses
        # (min-conflicts bias) with some randomness to allow escape.
        # Pick an unsatisfied clause, then a variable in it.
        # Efficient: maintain list of unsat clause indices.
        # To avoid O(m) scan each time, pick a random clause and retry a few times.
        clause_idx = None
        for _ in range(8):
            c = rng.randrange(m)
            if not clause_satisfied[c]:
                clause_idx = c
                break
        if clause_idx is None:
            # all satisfied handled above; fallback random
            clause_idx = rng.randrange(m)

        clause = clauses[clause_idx]
        # Choose a literal in the clause to flip its variable
        lit = clause[rng.randrange(len(clause))]
        v = abs(lit) - 1

        # Evaluate delta of flipping variable v
        # For each clause containing v, compute change in satisfied status
        delta = 0
        for ci in var_clauses[v]:
            pos = lit_count[ci][0]
            neg = lit_count[ci][1]
            cur_sat = (pos + neg) > 0
            # determine how v contributes
            # find v's sign in this clause
            # We need to know whether v appears positively or negatively.
            # We stored incidence without sign; recompute by scanning clause.
            # To keep it fast, precompute sign per (ci, v).
            # Instead, recompute: check literals.
            sign = 0
            for l in clauses[ci]:
                if abs(l) - 1 == v:
                    sign = 1 if l > 0 else -1
                    break
            if sign == 1:
                # v positive: currently contributes if assignment[v] True
                if assignment[v]:
                    new_pos = pos - 1
                    new_neg = neg
                else:
                    new_pos = pos + 1
                    new_neg = neg
            else:
                # v negative: contributes if assignment[v] False
                if not assignment[v]:
                    new_pos = pos
                    new_neg = neg - 1
                else:
                    new_pos = pos
                    new_neg = neg + 1
            new_sat = (new_pos + new_neg) > 0
            if cur_sat and not new_sat:
                delta += 1
            elif (not cur_sat) and new_sat:
                delta -= 1

        # Accept/reject
        if delta <= 0 or (T > 1e-9 and rng.random() < math.exp(-delta / T)):
            # apply flip
            assignment[v] = not assignment[v]
            for ci in var_clauses[v]:
                pos = lit_count[ci][0]
                neg = lit_count[ci][1]
                cur_sat = (pos + neg) > 0
                sign = 0
                for l in clauses[ci]:
                    if abs(l) - 1 == v:
                        sign = 1 if l > 0 else -1
                        break
                if sign == 1:
                    if assignment[v]:
                        new_pos = pos + 1
                        new_neg = neg
                    else:
                        new_pos = pos - 1
                        new_neg = neg
                else:
                    if not assignment[v]:
                        new_pos = pos
                        new_neg = neg + 1
                    else:
                        new_pos = pos
                        new_neg = neg - 1
                new_sat = (new_pos + new_neg) > 0
                lit_count[ci][0] = new_pos
                lit_count[ci][1] = new_neg
                clause_satisfied[ci] = new_sat
                if cur_sat and not new_sat:
                    total_unsat += 1
                elif (not cur_sat) and new_sat:
                    total_unsat -= 1

        T *= cooling
        step += 1

        if total_unsat < best_unsat:
            best_unsat = total_unsat
            best_assignment = assignment[:]
            no_improve = 0
        else:
            no_improve += 1

    # Final polish: greedy min-conflicts local search from best assignment
    if best_unsat > 0 and time.time() < deadline:
        assignment = best_assignment[:]
        # recompute
        clause_satisfied = [False] * m
        lit_count = [[0, 0] for _ in range(m)]
        for ci, clause in enumerate(clauses):
            pos = 0
            neg = 0
            for l in clause:
                if l > 0:
                    if assignment[abs(l) - 1]:
                        pos += 1
                else:
                    if not assignment[abs(l) - 1]:
                        neg += 1
            lit_count[ci][0] = pos
            lit_count[ci][1] = neg
            clause_satisfied[ci] = (pos + neg) > 0
        total_unsat = sum(1 for s in clause_satisfied if not s)
        best_unsat = total_unsat
        best_assignment = assignment[:]

        # greedy: repeatedly flip variable that reduces unsat most
        for _ in range(200000):
            if time.time() >= deadline:
                break
            if total_unsat == 0:
                break
            # find best single flip
            best_v = -1
            best_delta = 0
            # sample a subset of variables to keep it fast
            candidates = rng.sample(range(n), min(n, 200))
            for v in candidates:
                delta = 0
                for ci in var_clauses[v]:
                    pos = lit_count[ci][0]
                    neg = lit_count[ci][1]
                    cur_sat = (pos + neg) > 0
                    sign = 0
                    for l in clauses[ci]:
                        if abs(l) - 1 == v:
                            sign = 1 if l > 0 else -1
                            break
                    if sign == 1:
                        if assignment[v]:
                            new_pos = pos - 1
                            new_neg = neg
                        else:
                            new_pos = pos + 1
                            new_neg = neg
                    else:
                        if not assignment[v]:
                            new_pos = pos
                            new_neg = neg - 1
                        else:
                            new_pos = pos
                            new_neg = neg + 1
                    new_sat = (new_pos + new_neg) > 0
                    if cur_sat and not new_sat:
                        delta += 1
                    elif (not cur_sat) and new_sat:
                        delta -= 1
                if delta < best_delta:
                    best_delta = delta
                    best_v = v
            if best_v < 0:
                break
            v = best_v
            assignment[v] = not assignment[v]
            for ci in var_clauses[v]:
                pos = lit_count[ci][0]
                neg = lit_count[ci][1]
                cur_sat = (pos + neg) > 0
                sign = 0
                for l in clauses[ci]:
                    if abs(l) - 1 == v:
                        sign = 1 if l > 0 else -1
                        break
                if sign == 1:
                    if assignment[v]:
                        new_pos = pos + 1
                        new_neg = neg
                    else:
                        new_pos = pos - 1
                        new_neg = neg
                else:
                    if not assignment[v]:
                        new_pos = pos
                        new_neg = neg + 1
                    else:
                        new_pos = pos
                        new_neg = neg - 1
                new_sat = (new_pos + new_neg) > 0
                lit_count[ci][0] = new_pos
                lit_count[ci][1] = new_neg
                clause_satisfied[ci] = new_sat
                if cur_sat and not new_sat:
                    total_unsat += 1
                elif (not cur_sat) and new_sat:
                    total_unsat -= 1
            if total_unsat < best_unsat:
                best_unsat = total_unsat
                best_assignment = assignment[:]

    return best_assignment
# EVOLVE-BLOCK-END
