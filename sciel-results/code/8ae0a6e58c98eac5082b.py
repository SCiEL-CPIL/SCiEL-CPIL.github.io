"""Stochastic local search (simulated annealing + min-conflicts) with incremental
clause accounting, multiple restarts, and a final high-precision polish pass."""
import time
import math
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    start = time.time()
    DEADLINE = 290.0  # safety margin under the 300s kill

    # Precompute variable -> [(clause_index, sign)] for O(1) flip accounting.
    var_occ = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            var_occ[abs(lit) - 1].append((ci, 1 if lit > 0 else -1))

    def build_sat(assignment):
        # returns sat[] (bool per clause) and unsat_count
        sat = [False] * m
        uc = 0
        for ci, cl in enumerate(clauses):
            ok = False
            for lit in cl:
                if assignment[abs(lit) - 1] == (lit > 0):
                    ok = True
                    break
            sat[ci] = ok
            if not ok:
                uc += 1
        return sat, uc

    def is_solved(sat):
        return not any(sat) is False and not (False in sat)  # placeholder, replaced below

    best_assignment = None
    best_unsat = m + 1

    def try_record(assignment, uc):
        nonlocal best_assignment, best_unsat
        if uc < best_unsat:
            best_unsat = uc
            best_assignment = assignment[:]
        if uc == 0:
            return True
        return False

    def run_anneal(assignment, sat, uc, rng, budget, t0, t1):
        T = t0
        cool = (t1 / t0) ** (1.0 / max(budget, 1))
        for _ in range(budget):
            if uc == 0:
                break
            if time.time() >= DEADLINE:
                break
            # pick a random unsat clause
            ci = rng.randrange(m)
            while sat[ci]:
                ci = rng.randrange(m)
            cl = clauses[ci]
            # score each literal by how many other unsat clauses it would satisfy
            best_lit = None
            best_gain = -1
            for lit in cl:
                v = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                want = (lit > 0)
                if assignment[v] == want:
                    gain = 0
                else:
                    gain = 0
                    for cj, s in var_occ[v]:
                        if not sat[cj] and s != sign:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_lit = lit
            if best_lit is None:
                best_lit = cl[rng.randrange(len(cl))]
            # compute delta for flipping best_lit
            v = abs(best_lit) - 1
            sign = 1 if best_lit > 0 else -1
            delta = 0
            for cj, s in var_occ[v]:
                if sat[cj]:
                    if s == sign and all(
                        assignment[abs(l2) - 1] == (l2 > 0)
                        for l2 in clauses[cj]
                        if abs(l2) - 1 != v
                    ):
                        delta += 1
                else:
                    if s == sign:
                        delta -= 1
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                # apply flip
                assignment[v] = not assignment[v]
                for cj, s in var_occ[v]:
                    if sat[cj]:
                        if s == sign and all(
                            assignment[abs(l2) - 1] == (l2 > 0)
                            for l2 in clauses[cj]
                            if abs(l2) - 1 != v
                        ):
                            sat[cj] = False
                            uc += 1
                    else:
                        if s == sign:
                            sat[cj] = True
                            uc -= 1
            T *= cool
        return uc

    def run_minconflicts(assignment, sat, uc, rng, budget):
        for _ in range(budget):
            if uc == 0:
                break
            if time.time() >= DEADLINE:
                break
            ci = rng.randrange(m)
            while sat[ci]:
                ci = rng.randrange(m)
            cl = clauses[ci]
            # choose literal maximizing satisfied clauses
            best_lit = None
            best_gain = -1
            for lit in cl:
                v = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                if assignment[v] == (lit > 0):
                    gain = 0
                else:
                    gain = 0
                    for cj, s in var_occ[v]:
                        if not sat[cj] and s != sign:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_lit = lit
            if best_lit is None:
                best_lit = cl[rng.randrange(len(cl))]
            v = abs(best_lit) - 1
            sign = 1 if best_lit > 0 else -1
            assignment[v] = not assignment[v]
            for cj, s in var_occ[v]:
                if sat[cj]:
                    if s == sign and all(
                        assignment[abs(l2) - 1] == (l2 > 0)
                        for l2 in clauses[cj]
                        if abs(l2) - 1 != v
                    ):
                        sat[cj] = False
                        uc += 1
                else:
                    if s == sign:
                        sat[cj] = True
                        uc -= 1
        return uc

    def polish(assignment, budget):
        # high-precision deterministic-ish min-conflicts polish
        sat, uc = build_sat(assignment)
        if uc == 0:
            return assignment, uc
        rng = random.Random(12345)
        uc = run_minconflicts(assignment, sat, uc, rng, budget)
        return assignment, uc

    # Restart loop with alternating heuristics and fresh seeds
    restart = 0
    while time.time() < DEADLINE:
        seed = 1000 + restart
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat, uc = build_sat(assignment)
        if try_record(assignment, uc):
            break
        if restart % 2 == 0:
            # simulated annealing
            budget = 4000
            uc = run_anneal(assignment, sat, uc, rng, budget, t0=6.0, t1=0.05)
        else:
            # min-conflicts
            budget = 6000
            uc = run_minconflicts(assignment, sat, uc, rng, budget)
        try_record(assignment, uc)
        restart += 1
        if best_unsat == 0:
            break

    # Final high-precision polish on the best assignment found
    if best_assignment is not None:
        if time.time() < DEADLINE - 2.0:
            best_assignment, best_unsat = polish(best_assignment, budget=20000)

    return best_assignment if best_assignment is not None else [False] * n
