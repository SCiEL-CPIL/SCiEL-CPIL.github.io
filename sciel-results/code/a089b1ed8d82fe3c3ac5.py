"""
Hybrid 3-SAT solver: CDCL-lite (DPLL + watched literals + unit prop + pure lit)
plus a simulated-annealing / max-SAT portfolio with incremental clause-delta updates.
Structural change: replaces the greedy min-conflicts WalkSAT core with a
probabilistic SA landscape optimizer that tracks per-clause satisfaction counts
so each flip is O(occurrences) rather than O(clauses).
"""
import random
import time
import math
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]

    start_time = time.time()
    TIME_LIMIT = 290.0

    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)

    if time_remaining() <= 0:
        return [True] * n

    m = len(clauses)

    # Precompute clause data: list of (var_idx, is_positive)
    clause_data = []
    for c in clauses:
        cd = [(abs(l) - 1, l > 0) for l in c]
        clause_data.append(cd)

    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for i, cd in enumerate(clause_data):
        for var, pos in cd:
            var_to_clauses[var].append(i)

    def full_satisfied(assignment):
        for i in range(m):
            sat = False
            for var, pos in clause_data[i]:
                if assignment[var] == pos:
                    sat = True
                    break
            if not sat:
                return False
        return True

    # ---------- DPLL with watched literals ----------
    def dpll_solve():
        assignment = [-1] * n  # -1 unassigned, 0 False, 1 True

        clause_watches = []
        for i, c in enumerate(clauses):
            if len(c) >= 2:
                clause_watches.append([c[0], c[1]])
            elif len(c) == 1:
                clause_watches.append([c[0], c[0]])
            else:
                return None

        active_clauses = set(range(m))
        unit_queue = []

        def lit_sat(lit, val):
            if val == -1:
                return False
            return (val == 1) if lit > 0 else (val == 0)

        def lit_fals(lit, val):
            if val == -1:
                return False
            return (val == 0) if lit > 0 else (val == 1)

        def find_new_watch(ci, bad_lit):
            for lit in clauses[ci]:
                if lit != bad_lit:
                    v = abs(lit) - 1
                    if assignment[v] == -1:
                        return lit
            return None

        def propagate():
            while unit_queue:
                var_idx, val = unit_queue.pop()
                if assignment[var_idx] != -1 and assignment[var_idx] != val:
                    return False
                assignment[var_idx] = val

                for ci in var_to_clauses[var_idx]:
                    if ci not in active_clauses:
                        continue
                    w0, w1 = clause_watches[ci]
                    if lit_sat(w0, assignment[abs(w0) - 1]) or lit_sat(w1, assignment[abs(w1) - 1]):
                        continue
                    bad = None
                    other = None
                    if abs(w0) - 1 == var_idx:
                        bad, other = w0, w1
                    elif abs(w1) - 1 == var_idx:
                        bad, other = w1, w0
                    else:
                        sat = False
                        for lit in clauses[ci]:
                            v = abs(lit) - 1
                            if lit_sat(lit, assignment[v]):
                                sat = True
                                break
                        if sat:
                            active_clauses.discard(ci)
                        continue

                    if lit_fals(bad, val):
                        new_lit = find_new_watch(ci, bad)
                        if new_lit is None:
                            return False
                        if w0 == bad:
                            clause_watches[ci] = [new_lit, w1]
                        else:
                            clause_watches[ci] = [w0, new_lit]
                        nw0, nw1 = clause_watches[ci]
                        if lit_sat(nw0, assignment[abs(nw0) - 1]) or lit_sat(nw1, assignment[abs(nw1) - 1]):
                            active_clauses.discard(ci)
                        else:
                            n0_un = assignment[abs(nw0) - 1] == -1
                            n1_un = assignment[abs(nw1) - 1] == -1
                            n0_f = lit_fals(nw0, assignment[abs(nw0) - 1])
                            n1_f = lit_fals(nw1, assignment[abs(nw1) - 1])
                            if n0_f and n1_un:
                                unit_queue.append((abs(nw1) - 1, 1 if nw1 > 0 else 0))
                            elif n1_f and n0_un:
                                unit_queue.append((abs(nw0) - 1, 1 if nw0 > 0 else 0))
                    else:
                        active_clauses.discard(ci)
            return True

        # Initial unit clauses
        for i in range(m):
            if i not in active_clauses:
                continue
            sat = False
            for lit in clauses[i]:
                v = abs(lit) - 1
                if lit_sat(lit, assignment[v]):
                    sat = True
                    break
            if sat:
                active_clauses.discard(i)
                continue
            unassigned_lits = []
            for lit in clauses[i]:
                v = abs(lit) - 1
                if assignment[v] == -1:
                    unassigned_lits.append(lit)
            if len(unassigned_lits) == 1:
                lit = unassigned_lits[0]
                var_idx = abs(lit) - 1
                val = 1 if lit > 0 else 0
                if assignment[var_idx] == -1:
                    unit_queue.append((var_idx, val))

        if not propagate():
            return None

        for i in range(m):
            if i in active_clauses:
                sat = False
                for lit in clauses[i]:
                    v = abs(lit) - 1
                    if lit_sat(lit, assignment[v]):
                        sat = True
                        break
                if not sat:
                    break
        else:
            return [bool(a) for a in assignment]

        # Pure literal elimination
        def eliminate_pure():
            pure_assignments = []
            for var_idx in range(n):
                if assignment[var_idx] != -1:
                    continue
                has_pos = False
                has_neg = False
                for ci in var_to_clauses[var_idx]:
                    if ci not in active_clauses:
                        continue
                    for lit in clauses[ci]:
                        if abs(lit) - 1 == var_idx:
                            if lit > 0:
                                has_pos = True
                            else:
                                has_neg = True
                            break
                if has_pos and not has_neg:
                    pure_assignments.append((var_idx, 1))
                elif has_neg and not has_pos:
                    pure_assignments.append((var_idx, 0))
            for var_idx, val in pure_assignments:
                if assignment[var_idx] != -1:
                    if assignment[var_idx] != val:
                        return None
                    continue
                assignment[var_idx] = val
                unit_queue.append((var_idx, val))
            return pure_assignments

        while True:
            if time_remaining() < 5.0:
                break
            pl = eliminate_pure()
            if pl is None:
                return None
            if not pl:
                break
            if not propagate():
                return None

        def dpll_rec(depth):
            if time_remaining() < 5.0:
                return None
            all_sat = True
            for i in range(m):
                if i not in active_clauses:
                    continue
                sat = False
                for lit in clauses[i]:
                    v = abs(lit) - 1
                    if lit_sat(lit, assignment[v]):
                        sat = True
                        break
                if not sat:
                    all_sat = False
                    break
            if all_sat:
                return [bool(a) for a in assignment]

            var_counts = [0] * n
            for i in active_clauses:
                for lit in clauses[i]:
                    v = abs(lit) - 1
                    if assignment[v] == -1:
                        var_counts[v] += 1

            best_var = -1
            best_count = -1
            for v in range(n):
                if assignment[v] == -1 and var_counts[v] > best_count:
                    best_count = var_counts[v]
                    best_var = v

            if best_var == -1:
                return None

            for val in [1, 0]:
                if time_remaining() < 5.0:
                    return None
                saved_assignment = assignment[:]
                saved_active = active_clauses.copy()
                saved_watches = [w[:] for w in clause_watches]
                saved_queue = unit_queue[:]

                assignment[best_var] = val
                unit_queue.append((best_var, val))

                if propagate():
                    result = dpll_rec(depth + 1)
                    if result is not None:
                        return result

                assignment[:] = saved_assignment
                active_clauses.clear()
                active_clauses.update(saved_active)
                for i in range(m):
                    clause_watches[i] = saved_watches[i][:]
                unit_queue[:] = saved_queue

            return None

        return dpll_rec(0)

    dpll_result = dpll_solve()
    if dpll_result is not None and full_satisfied(dpll_result):
        return dpll_result

    # ---------- Simulated Annealing Max-SAT ----------
    # Precompute per-variable occurrence lists: (clause_idx, is_positive)
    var_occ = [[] for _ in range(n)]
    for i, cd in enumerate(clause_data):
        for var, pos in cd:
            var_occ[var].append((i, pos))

    def initial_satisfaction_counts(assignment):
        """Return list of satisfaction counts per clause and total satisfied."""
        counts = [0] * m
        total = 0
        for i, cd in enumerate(clause_data):
            cnt = 0
            for var, pos in cd:
                if assignment[var] == pos:
                    cnt += 1
            counts[i] = cnt
            if cnt > 0:
                total += 1
        return counts, total

    def simulated_annealing(seed, max_steps, t0=10.0, cooling=0.9997, min_t=0.05):
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        counts, total_sat = initial_satisfaction_counts(assignment)
        best_total = total_sat
        best_assignment = assignment[:]

        if total_sat == m:
            return assignment[:]

        temp = t0
        for step in range(max_steps):
            if time_remaining() < 2.0:
                break

            # Pick a variable to flip: bias toward variables in unsatisfied clauses
            unsat_clauses = [i for i in range(m) if counts[i] == 0]
            if not unsat_clauses:
                return assignment[:]

            # Choose an unsatisfied clause, then a variable in it
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            var, pos = clause_data[ci][rng.randrange(3)]

            # Compute delta in satisfied clause count
            delta = 0
            for (oci, opos) in var_occ[var]:
                if counts[oci] == 0:
                    # Clause unsat; flipping var will satisfy it iff new value matches
                    if (not assignment[var]) == opos:
                        delta += 1
                elif counts[oci] == 1:
                    # Clause has exactly one satisfying literal; if it's this var, flipping breaks it
                    if assignment[var] == opos:
                        delta -= 1

            # Accept with SA probability
            if delta >= 0:
                accept = True
            else:
                if temp < min_t:
                    accept = False
                else:
                    accept = rng.random() < math.exp(delta / temp)

            if accept:
                assignment[var] = not assignment[var]
                # Update counts
                for (oci, opos) in var_occ[var]:
                    old_cnt = counts[oci]
                    if assignment[var] == opos:
                        counts[oci] = old_cnt + 1
                    else:
                        counts[oci] = old_cnt - 1
                    if old_cnt == 0 and counts[oci] > 0:
                        total_sat += 1
                    elif old_cnt > 0 and counts[oci] == 0:
                        total_sat -= 1

                if total_sat > best_total:
                    best_total = total_sat
                    best_assignment = assignment[:]

                if total_sat == m:
                    return assignment[:]

            temp *= cooling

        return best_assignment

    def count_unsat(assignment):
        cnt = 0
        for i in range(m):
            sat = False
            for var, pos in clause_data[i]:
                if assignment[var] == pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    best_result = None
    best_unsat_count = float('inf')

    def try_sa(seed, max_steps, t0=10.0, cooling=0.9997):
        nonlocal best_result, best_unsat_count
        if time_remaining() < 5.0:
            return None
        result = simulated_annealing(seed, max_steps, t0=t0, cooling=cooling)
        if result is not None:
            if full_satisfied(result):
                return result
            else:
                cnt = count_unsat(result)
                if cnt < best_unsat_count:
                    best_unsat_count = cnt
                    best_result = result[:]
        return None

    # SA portfolio with diverse parameters and seeds
    # Phase 1: longer runs with moderate cooling
    for seed in [42, 123, 777, 2024, 555, 999, 31337, 12345, 67890, 24680]:
        if time_remaining() < 5.0:
            break
        res = try_sa(seed, max(2000, n * 60), t0=10.0, cooling=0.9995)
        if res is not None:
            return res

    # Phase 2: shorter runs with faster cooling (more exploitation)
    if time_remaining() > 10.0:
        for seed in [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]:
            if time_remaining() < 5.0:
                break
            res = try_sa(seed, max(1000, n * 30), t0=5.0, cooling=0.999)
            if res is not None:
                return res

    # Phase 3: high-temperature exploration
    if time_remaining() > 10.0:
        for seed in [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]:
            if time_remaining() < 5.0:
                break
            res = try_sa(seed, max(1500, n * 45), t0=20.0, cooling=0.9998)
            if res is not None:
                return res

    # Phase 4: many short restarts
    if time_remaining() > 5.0:
        for seed in range(2000, 2050):
            if time_remaining() < 3.0:
                break
            res = try_sa(seed, 500, t0=8.0, cooling=0.9993)
            if res is not None:
                return res

    if best_result is not None:
        return best_result

    return [True] * n
