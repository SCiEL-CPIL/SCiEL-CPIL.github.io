"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    deadline = time.time() + 310.0

    # Preprocess
    cvars = []
    csigns = []
    for cl in clauses:
        vs = [abs(l) - 1 for l in cl]
        ss = [l > 0 for l in cl]
        cvars.append(vs)
        csigns.append(ss)

    # var -> list of (clause_idx, position_in_clause)
    var_clauses = [[] for _ in range(n)]
    for ci, vs in enumerate(cvars):
        for pos, v in enumerate(vs):
            var_clauses[v].append((ci, pos))

    rng = random.Random(42)

    def check_all(assignment):
        unsat = []
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            sat = False
            for j in range(len(vs)):
                if assignment[vs[j]] == ss[j]:
                    sat = True
                    break
            if not sat:
                unsat.append(ci)
        return unsat

    def walk_sat(assignment, max_flips):
        # Initialize sat_count
        sat_count = [0] * m
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            cnt = 0
            for j in range(len(vs)):
                if assignment[vs[j]] == ss[j]:
                    cnt += 1
            sat_count[ci] = cnt

        unsat_set = set(ci for ci in range(m) if sat_count[ci] == 0)

        if not unsat_set:
            return assignment, []

        unsat_list = list(unsat_set)

        for _ in range(max_flips):
            if not unsat_list:
                return assignment, []

            if time.time() > deadline:
                break

            ci = rng.choice(unsat_list)
            vs = cvars[ci]
            ss = csigns[ci]

            # Find false literals in this clause
            false_positions = []
            for j in range(len(vs)):
                if not (assignment[vs[j]] == ss[j]):
                    false_positions.append(j)

            if false_positions:
                if rng.random() < 0.8:
                    j = rng.choice(false_positions)
                else:
                    j = rng.randrange(len(vs))
            else:
                j = rng.randrange(len(vs))

            v = vs[j]
            assignment[v] = not assignment[v]

            # Update all clauses containing v
            new_unsat = []
            for ci2, pos2 in var_clauses[v]:
                sign = csigns[ci2][pos2]
                if assignment[v] == sign:
                    sat_count[ci2] += 1
                    if sat_count[ci2] == 1:
                        # Was unsat, now sat
                        if ci2 in unsat_set:
                            unsat_set.discard(ci2)
                else:
                    sat_count[ci2] -= 1
                    if sat_count[ci2] == 0:
                        # Was sat, now unsat
                        if ci2 not in unsat_set:
                            unsat_set.add(ci2)

            # Rebuild unsat_list periodically or when changed significantly
            # For efficiency, rebuild every 100 flips or when unsat_set changed
            # Simple approach: rebuild each iteration (acceptable for small unsat)
            unsat_list = list(unsat_set)

        return assignment, list(unsat_set)

    best_unsat_count = float('inf')
    best_assignment = None

    restarts = 0
    base_flips = 1000
    while time.time() < deadline:
        restarts += 1
        flip_budget = base_flips + restarts * 500
        flip_budget = min(flip_budget, 100000)

        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat = check_all(assignment)

        if not unsat:
            return assignment

        if len(unsat) < best_unsat_count:
            best_unsat_count = len(unsat)
            best_assignment = assignment[:]

        assignment, unsat = walk_sat(assignment, flip_budget)

        if not unsat:
            return assignment

        if len(unsat) < best_unsat_count:
            best_unsat_count = len(unsat)
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
