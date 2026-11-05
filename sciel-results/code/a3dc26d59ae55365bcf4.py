"""Planted 3-SAT solver (50 vars) using fast incremental Max-SAT local search."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses_in = instance["clauses"]

    # Deadline guard: return best found with a safety margin before the kill.
    deadline = time.monotonic() + 280.0

    # Precompute per-literal variable index and clause polarity.
    # clauses: list of (var, sign) with var in 0..n-1, sign in {+1,-1}
    clauses = []
    for c in clauses_in:
        clauses.append(tuple((abs(l) - 1, 1 if l > 0 else -1) for l in c))
    m = len(clauses)

    # Per-clause satisfied-literal count (0..len). 0 means unsatisfied.
    def clause_ok(cl, assign):
        cnt = 0
        for v, s in cl:
            if assign[v] == (s > 0):
                cnt += 1
        return cnt

    # Bookkeeping for fast flips.
    # For each clause, we need to know, for each variable in it, whether that
    # variable currently satisfies the clause. We store per-clause a small list.
    # To make flips O(1), maintain:
    #   clause_sat[ci] = number of satisfied literals in clause ci
    #   var_clause_sat[v] = number of clauses where v is a satisfying literal
    # Flipping v: for each clause ci containing v, let was = (v satisfied ci).
    #   clause_sat[ci] += -1 if was else +1
    #   var_clause_sat[v] changes by (+1 if new_satisfies else -1 if was else 0)

    def build_state(assign):
        clause_sat = [0] * m
        var_clause_sat = [0] * n
        # sat_lit[ci] = list of booleans parallel to clauses[ci]
        sat_lit = []
        for ci, cl in enumerate(clauses):
            cnt = 0
            sl = []
            for v, s in cl:
                ok = assign[v] == (s > 0)
                sl.append(ok)
                if ok:
                    cnt += 1
            sat_lit.append(sl)
            clause_sat[ci] = cnt
            for (v, s), ok in zip(cl, sl):
                if ok:
                    var_clause_sat[v] += 1
        return clause_sat, var_clause_sat, sat_lit

    def flip(v, assign, clause_sat, var_clause_sat, sat_lit):
        # For each clause containing v, update counts.
        # We need the list of (ci, index-in-clause) for variable v.
        for ci, idx in var_clause_map[v]:
            was = sat_lit[ci][idx]
            # new satisfaction = not was (since v's value flips, its literal flips)
            if was:
                clause_sat[ci] -= 1
                var_clause_sat[v] -= 1
            else:
                clause_sat[ci] += 1
                var_clause_sat[v] += 1
            sat_lit[ci][idx] = not was
        assign[v] = not assign[v]

    # Build variable -> [(clause index, position in clause)] map.
    var_clause_map = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for idx, (v, s) in enumerate(cl):
            var_clause_map[v].append((ci, idx))

    def unsat_count(clause_sat):
        return m - sum(1 for x in clause_sat if x > 0)

    rng = random.Random(0)

    best_assign = None
    best_unsat = m + 1

    def try_assign(assign):
        nonlocal best_assign, best_unsat
        clause_sat, var_clause_sat, sat_lit = build_state(assign)
        uc = unsat_count(clause_sat)
        if uc < best_unsat:
            best_unsat = uc
            best_assign = assign[:]
        if uc == 0:
            return assign, clause_sat, var_clause_sat, sat_lit, True
        return assign, clause_sat, var_clause_sat, sat_lit, False

    # Local search: WalkSAT with min-conflict variable heuristic + restarts.
    max_steps = 40000
    restart_budget = 2000
    t0 = time.monotonic()

    while True:
        assign = [rng.random() < 0.5 for _ in range(n)]
        clause_sat, var_clause_sat, sat_lit = build_state(assign)
        uc = unsat_count(clause_sat)
        if uc < best_unsat:
            best_unsat = uc
            best_assign = assign[:]
        if uc == 0:
            break
        steps = 0
        while steps < max_steps:
            if uc == 0:
                break
            # Pick a random unsatisfied clause.
            # For speed, gather unsat clause indices lazily.
            # Find an unsat clause.
            # Use a random probe; if satisfied, try again a few times.
            ci = rng.randrange(m)
            if clause_sat[ci] > 0:
                # find an actually unsat one by scanning from ci
                found = False
                for k in range(m):
                    j = (ci + k) % m
                    if clause_sat[j] == 0:
                        ci = j
                        found = True
                        break
                if not found:
                    break
            cl = clauses[ci]
            # Min-conflict: pick variable in this clause whose flip reduces
            # unsat count the most. Compute delta = (# unsat clauses fixed) -
            # (# sat clauses broken) for flipping v, restricted to clauses
            # containing v. We approximate using var_clause_sat.
            # A flip of v changes, for each clause containing v:
            #   if v currently satisfies it: that clause may become unsat (if it
            #   had exactly 1 sat literal) -> +1 unsat
            #   if v currently unsatisfies it: that clause gains a sat literal
            #   -> -1 unsat (it was already unsat)
            # Net unsat change = (was_satisfying and clause_sat==1 ? +1 : 0)
            #                    + (not was_satisfying ? -1 : 0)
            # We want to minimise that (most negative = best).
            best_v = None
            best_delta = None
            for idx, (v, s) in enumerate(cl):
                was = sat_lit[ci][idx]
                delta = 0
                # consider all clauses containing v
                for cj, _ in var_clause_map[v]:
                    if cj == ci:
                        # handled below via was
                        if was and clause_sat[cj] == 1:
                            delta += 1
                        elif not was:
                            delta -= 1
                    else:
                        # need whether v satisfies clause cj
                        # find position
                        # (rare, small clauses) -> compute directly
                        vpos = None
                        for p, (vv, ss) in enumerate(clauses[cj]):
                            if vv == v:
                                vpos = p
                                break
                        wasj = sat_lit[cj][vpos]
                        if wasj and clause_sat[cj] == 1:
                            delta += 1
                        elif not wasj:
                            delta -= 1
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_v = v
            # Random tie-breaking among near-best to escape plateaus:
            # with small probability pick a random variable in the clause.
            if rng.random() < 0.2:
                best_v = cl[rng.randrange(len(cl))][0]
            flip(best_v, assign, clause_sat, var_clause_sat, sat_lit)
            uc = unsat_count(clause_sat)
            if uc < best_unsat:
                best_unsat = uc
                best_assign = assign[:]
            steps += 1
        if best_unsat == 0:
            break
        # Restart check
        if time.monotonic() > deadline:
            break
        restart_budget -= 1
        if restart_budget <= 0:
            break
    if best_assign is None:
        best_assign = assign
    return best_assign
# EVOLVE-BLOCK-END
