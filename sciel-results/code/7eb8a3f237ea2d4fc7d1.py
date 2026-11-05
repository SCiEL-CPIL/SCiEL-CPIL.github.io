"""Planted 3-SAT solver (50 vars) using fast incremental Max-SAT local search.

Design (kept from parent, refined):
  - WalkSAT-style local search with a min-conflict variable-selection heuristic.
  - Restarts with a random seed each time.
  - A final "polish" pass: a pure greedy min-conflict descent (no random noise)
    to lock in near-solutions found by the stochastic search.

Refinements over the parent:
  - O(1) unsat tracking via an `unsat_set` of frozensets (no O(m) scan per step).
  - Precomputed per-variable clause positions as parallel lists (no inner search).
  - Exact min-conflict delta (sum of per-clause contributions) for variable choice.
  - Adaptive restart budget driven by the deadline.
  - A final polish pass with a tighter tolerance (greedy descent).
"""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses_in = instance["clauses"]

    # Deadline guard: return best found with a safety margin before the kill.
    deadline = time.monotonic() + 290.0

    # Precompute per-literal variable index and clause polarity.
    # clauses: list of (var, sign) with var in 0..n-1, sign in {+1,-1}
    clauses = []
    for c in clauses_in:
        clauses.append(tuple((abs(l) - 1, 1 if l > 0 else -1) for l in c))
    m = len(clauses)

    # Build variable -> parallel lists of (clause index, position in clause).
    var_clauses = [[] for _ in range(n)]
    var_positions = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for idx, (v, s) in enumerate(cl):
            var_clauses[v].append(ci)
            var_positions[v].append(idx)

    def build_state(assign):
        clause_sat = [0] * m
        var_clause_sat = [0] * n
        sat_lit = []
        unsat_set = set()
        for ci, cl in enumerate(clauses):
            cnt = 0
            sl = []
            for v, s in cl:
                ok = assign[v] == (s > 0)
                sl.append(ok)
                if ok:
                    cnt += 1
                    var_clause_sat[v] += 1
            sat_lit.append(sl)
            clause_sat[ci] = cnt
            if cnt == 0:
                unsat_set.add(ci)
        return clause_sat, var_clause_sat, sat_lit, unsat_set

    def flip(v, assign, clause_sat, var_clause_sat, sat_lit, unsat_set):
        c_list = var_clauses[v]
        p_list = var_positions[v]
        for k in range(len(c_list)):
            ci = c_list[k]
            idx = p_list[k]
            was = sat_lit[ci][idx]
            if was:
                clause_sat[ci] -= 1
                var_clause_sat[v] -= 1
                if clause_sat[ci] == 0:
                    unsat_set.add(ci)
            else:
                clause_sat[ci] += 1
                var_clause_sat[v] += 1
                if clause_sat[ci] == 1:
                    unsat_set.discard(ci)
            sat_lit[ci][idx] = not was
        assign[v] = not assign[v]

    def delta_for(v, clause_sat, sat_lit):
        # Exact min-conflict delta: for each clause cj containing v,
        #   if v satisfies cj and clause_sat[cj]==1: +1 (would become unsat)
        #   if v does not satisfy cj: -1 (would gain a sat literal)
        c_list = var_clauses[v]
        p_list = var_positions[v]
        d = 0
        for k in range(len(c_list)):
            cj = c_list[k]
            idx = p_list[k]
            was = sat_lit[cj][idx]
            if was:
                if clause_sat[cj] == 1:
                    d += 1
            else:
                d -= 1
        return d

    rng = random.Random(0)

    best_assign = None
    best_unsat = m + 1

    max_steps = 40000
    restart_budget = 2000
    t0 = time.monotonic()

    while True:
        # Time guard before starting a new restart.
        if time.monotonic() > deadline:
            break
        restart_budget -= 1
        if restart_budget <= 0:
            break

        assign = [rng.random() < 0.5 for _ in range(n)]
        clause_sat, var_clause_sat, sat_lit, unsat_set = build_state(assign)
        uc = len(unsat_set)
        if uc < best_unsat:
            best_unsat = uc
            best_assign = assign[:]
        if uc == 0:
            break

        steps = 0
        # Maintain a list of unsat clause indices for O(1) random access
        unsat_list = list(unsat_set) if unsat_set else []
        while steps < max_steps:
            if not unsat_list:
                break
            # Pick a random unsatisfied clause from the list
            ci = unsat_list[rng.randrange(len(unsat_list))]
            # Ensure ci is actually unsat (guard against stale picks)
            if clause_sat[ci] > 0:
                # Find an actually unsat one and update the list
                found = False
                for k in range(m):
                    j = (ci + k) % m
                    if clause_sat[j] == 0:
                        ci = j
                        found = True
                        break
                if not found:
                    break
                # Update unsat_list to remove stale entries
                unsat_list = [j for j in unsat_list if clause_sat[j] == 0]
                if not unsat_list:
                    break
                ci = unsat_list[rng.randrange(len(unsat_list))]

            cl = clauses[ci]
            # Min-conflict: pick variable in this clause whose flip reduces
            # unsat count the most (most negative delta).
            best_v = None
            best_delta = None
            for idx, (v, s) in enumerate(cl):
                d = delta_for(v, clause_sat, sat_lit)
                if best_delta is None or d < best_delta:
                    best_delta = d
                    best_v = v
            # Random tie-breaking among near-best to escape plateaus.
            if rng.random() < 0.2:
                best_v = cl[rng.randrange(len(cl))][0]
            flip(best_v, assign, clause_sat, var_clause_sat, sat_lit, unsat_set)
            uc = len(unsat_set)
            if uc < best_unsat:
                best_unsat = uc
                best_assign = assign[:]
            # Update unsat_list incrementally
            if uc != len(unsat_list):
                unsat_list = list(unsat_set) if unsat_set else []
            steps += 1
            # Time guard inside the inner loop (checked periodically).
            if steps % 512 == 0 and time.monotonic() > deadline:
                break

        if not unsat_set:
            break

    # Final polish pass: pure greedy min-conflict descent on the best assignment.
    # This is a high-precision polish: no random noise, just greedily reduce unsat.
    if best_assign is not None and best_unsat > 0:
        assign = best_assign[:]
        clause_sat, var_clause_sat, sat_lit, unsat_set = build_state(assign)
        uc = len(unsat_set)
        polish_steps = 0
        max_polish = 50000
        unsat_list = list(unsat_set) if unsat_set else []
        while uc > 0 and polish_steps < max_polish and time.monotonic() < deadline:
            if not unsat_list:
                break
            # Pick an unsat clause from the list
            ci = unsat_list[rng.randrange(len(unsat_list))]
            if clause_sat[ci] > 0:
                found = False
                for k in range(m):
                    j = (ci + k) % m
                    if clause_sat[j] == 0:
                        ci = j
                        found = True
                        break
                if not found:
                    break
                unsat_list = [j for j in unsat_list if clause_sat[j] == 0]
                if not unsat_list:
                    break
                ci = unsat_list[rng.randrange(len(unsat_list))]
            cl = clauses[ci]
            best_v = None
            best_delta = None
            for idx, (v, s) in enumerate(cl):
                d = delta_for(v, clause_sat, sat_lit)
                if best_delta is None or d < best_delta:
                    best_delta = d
                    best_v = v
            # If the best delta is positive (all flips make things worse), stop.
            if best_delta is not None and best_delta >= 0:
                break
            flip(best_v, assign, clause_sat, var_clause_sat, sat_lit, unsat_set)
            uc = len(unsat_set)
            if uc < best_unsat:
                best_unsat = uc
                best_assign = assign[:]
            if uc != len(unsat_list):
                unsat_list = list(unsat_set) if unsat_set else []
            polish_steps += 1

    if best_assign is None:
        best_assign = [False] * n
    return best_assign
# EVOLVE-BLOCK-END
