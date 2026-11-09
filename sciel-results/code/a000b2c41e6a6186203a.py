# EVOLVE-BLOCK-START
import time
import random
import heapq


def _satisfied(cl, ass):
    for v, p in cl:
        if ass[v] == p:
            return True
    return False


def _dpll_solve(clauses_opt, var_clauses, n, m, deadline):
    assign = [-1] * n
    trail = []
    level = [0] * n
    learnt = []
    
    activity = [1.0] * n
    var_heap = [(-activity[v], v) for v in range(n)]
    heapq.heapify(var_heap)
    phase = [True] * n
    
    def propagate(queue):
        while queue:
            ci, is_learnt = queue.pop()
            if is_learnt:
                if ci >= len(learnt): continue
                cl = learnt[ci]
            else:
                if ci >= m: continue
                cl = clauses_opt[ci]
            
            satisfied = False
            unfixed = []
            for v, p in cl:
                if assign[v] != -1:
                    if assign[v] == (1 if p else 0):
                        satisfied = True
                        break
                else:
                    unfixed.append((v, p))
            
            if satisfied:
                continue
            if not unfixed:
                return False, (ci, is_learnt)
            if len(unfixed) == 1:
                v, p = unfixed[0]
                if assign[v] == -1:
                    assign[v] = 1 if p else 0
                    trail.append(v)
                    level[v] = len(trail)
                    for ci2, p2 in var_clauses[v]:
                        queue.append((ci2, False))
                    for li in range(len(learnt)):
                        for v2, p2 in learnt[li]:
                            if v2 == v:
                                queue.append((li, True))
                                break
        return True, None

    def analyze_conflict(conf_clause):
        ci, is_learnt = conf_clause
        if is_learnt:
            cl = learnt[ci]
        else:
            cl = clauses_opt[ci]
        
        learnt_clause = []
        max_level = 0
        for v, p in cl:
            if assign[v] != -1:
                learnt_clause.append((v, p))
                if level[v] > max_level:
                    max_level = level[v]
        
        if len(learnt_clause) > 1:
            seen = set()
            unique = []
            for v, p in learnt_clause:
                if (v, p) not in seen:
                    seen.add((v, p))
                    unique.append((v, p))
            learnt.append(unique)
            for v, p in unique:
                activity[v] += 1.0
            return unique, max_level
        return None, 0

    q = [(ci, False) for ci in range(m)]
    ok, conf = propagate(q)
    if not ok:
        return None

    while time.time() < deadline:
        v = -1
        while var_heap:
            neg_act, cand = var_heap[0]
            if assign[cand] == -1:
                v = cand
                heapq.heappop(var_heap)
                break
            else:
                heapq.heappop(var_heap)
        
        if v == -1:
            for ci in range(m):
                cl = clauses_opt[ci]
                for vi, pi in cl:
                    if assign[vi] == (1 if pi else 0):
                        break
                else:
                    return None
            return [bool(assign[i]) for i in range(n)]
        
        val = 1 if phase[v] else 0
        assign[v] = val
        trail.append(v)
        level[v] = len(trail)
        
        q = [(ci2, False) for ci2, p2 in var_clauses[v]]
        for li in range(len(learnt)):
            for v2, p2 in learnt[li]:
                if v2 == v:
                    q.append((li, True))
                    break
        
        ok, conf = propagate(q)
        
        if ok:
            solved = True
            for ci in range(m):
                cl = clauses_opt[ci]
                for vi, pi in cl:
                    if assign[vi] == (1 if pi else 0):
                        break
                else:
                    solved = False
                    break
            if solved:
                return [bool(assign[i]) for i in range(n)]
        else:
            learnt_clause, bt_level = analyze_conflict(conf)
            while trail and level[trail[-1]] >= bt_level:
                vv = trail.pop()
                assign[vv] = -1
                level[vv] = 0
            if not trail:
                return None
        
        val2 = 1 - val
        assign[v] = val2
        trail.append(v)
        level[v] = len(trail)
        phase[v] = not phase[v]
        
        q = [(ci2, False) for ci2, p2 in var_clauses[v]]
        for li in range(len(learnt)):
            for v2, p2 in learnt[li]:
                if v2 == v:
                    q.append((li, True))
                    break
        
        ok, conf = propagate(q)
        
        if ok:
            solved = True
            for ci in range(m):
                cl = clauses_opt[ci]
                for vi, pi in cl:
                    if assign[vi] == (1 if pi else 0):
                        break
                else:
                    solved = False
                    break
            if solved:
                return [bool(assign[i]) for i in range(n)]
        else:
            learnt_clause, bt_level = analyze_conflict(conf)
            while trail and level[trail[-1]] >= bt_level:
                vv = trail.pop()
                assign[vv] = -1
                level[vv] = 0
            if not trail:
                return None

    return None


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 320  # safety margin before 330s kill

    # Precompute clauses as (var_idx 0-based, polarity bool)
    clauses_opt = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_opt)

    # Precompute var -> list of (clause_idx, polarity)
    var_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses_opt):
        for v, p in cl:
            var_clauses[v].append((ci, p))

    # Weights per variable
    weights = [len(l) for l in var_clauses]
    total_w = sum(weights) or 1

    # Phase 0: Try DPLL with clause learning
    dpll_deadline = start_time + 20  # Give DPLL 20 seconds
    result = _dpll_solve(clauses_opt, var_clauses, n, m, dpll_deadline)
    if result is not None:
        # Verify
        for cl in clauses_opt:
            for v, p in cl:
                if result[v] == p:
                    break
            else:
                result = None
                break
        if result is not None:
            return result

    # Fallback to SLS if DPLL didn't solve
    def count_unsat(ass):
        cnt = 0
        for cl in clauses_opt:
            for v, p in cl:
                if ass[v] == p:
                    break
            else:
                cnt += 1
        return cnt

    def build_state(ass):
        # sat[v] = number of satisfied literals in clause v
        sat = [0] * m
        for ci, cl in enumerate(clauses_opt):
            s = 0
            for v, p in cl:
                if ass[v] == p:
                    s += 1
            sat[ci] = s
        return sat

    def flip_var(ass, var, sat):
        old = ass[var]
        new = not old
        ass[var] = new
        for ci, p in var_clauses[var]:
            if p == new:
                sat[ci] += 1
            else:
                sat[ci] -= 1

    def check_all_sat(sat):
        for s in sat:
            if s == 0:
                return False
        return True

    def unsat_clauses_list(sat):
        return [ci for ci in range(m) if sat[ci] == 0]

    def unit_propagate(ass, remaining_vars, n_remaining, sat):
        # Repeat: find clauses with exactly one unfixed unsat literal.
        # Returns True if no contradiction (ass updated in place).
        # Uses remaining_vars set.
        changed = True
        while changed and n_remaining > 0:
            changed = False
            # For each clause still unsat, count unfixed literals
            for ci in range(m):
                if sat[ci] > 0:
                    continue
                cl = clauses_opt[ci]
                unfixed = []
                for v, p in cl:
                    if v not in remaining_vars:
                        # fixed: if it's satisfied, clause would be sat; but sat==0 so it's unsat
                        continue
                    unfixed.append((v, p))
                if len(unfixed) == 1:
                    v, p = unfixed[0]
                    if ass[v] != p:
                        flip_var(ass, v, sat)
                    remaining_vars.discard(v)
                    n_remaining -= 1
                    changed = True
        return True  # no contradiction handling needed (planted SAT)

    # Phase 1: unit propagation on a random seed to get a good starting point
    best_assignment = None
    best_unsat = float('inf')

    def track(ass):
        nonlocal best_assignment, best_unsat
        u = count_unsat(ass)
        if u < best_unsat:
            best_unsat = u
            best_assignment = ass[:]
        return u

    def quick_check(ass):
        for cl in clauses_opt:
            for v, p in cl:
                if ass[v] == p:
                    break
            else:
                return False
        return True

    max_restarts = 200
    rng_master = random.Random(123456789)

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(restart * 7919 + 131)
        ass = [rng.random() < 0.5 for _ in range(n)]

        # Try to bias toward positive literals (planted instances often)
        # Occasionally use phase bias: prefer the literal that appears more
        if restart % 5 == 0:
            # phase save bias: set var to polarity with more occurrences
            for v in range(n):
                pos = sum(1 for ci, p in var_clauses[v] if p)
                ass[v] = (pos >= len(var_clauses[v]) - pos) if len(var_clauses[v]) else False

        u = track(ass)
        if u == 0:
            return ass

        # Unit propagation to reduce
        remaining_vars = set(range(n))
        n_remaining = n
        sat = build_state(ass)
        unit_propagate(ass, remaining_vars, n_remaining, sat)
        u = track(ass)
        if u == 0:
            return ass

        # Determine heuristic mode for this restart
        mode = restart % 3  # 0: min-break, 1: phase-satisfy, 2: max-weight
        noise = 0.2 + 0.1 * (restart % 4)

        steps = 20000
        for step in range(steps):
            if time.time() > deadline:
                break

            # Find an unsatisfied clause - use longest unsat for stronger guidance
            best_ci = -1
            best_len = 0
            # Sample to find a long unsat clause quickly
            # Check all but stop early is expensive; do a partial scan
            # We'll scan but only track; m is moderate.
            for ci in range(m):
                if sat[ci] == 0:
                    ln = len(clauses_opt[ci])
                    if ln > best_len:
                        best_len = ln
                        best_ci = ci
                        if ln >= 3:
                            # 3 is max for 3-SAT, can break
                            break
            if best_ci < 0:
                return ass

            clause = clauses_opt[best_ci]

            # Choose variable
            if rng.random() < noise:
                v, p = clause[rng.randrange(len(clause))]
            else:
                # Evaluate each literal
                best_v = clause[0][0]
                best_score = None
                for v, p in clause:
                    # new value would be p (to satisfy this literal)
                    if ass[v] == p:
                        # already satisfied? shouldn't be in unsat clause but be safe
                        continue
                    # cost = number of currently-satisfied other clauses broken
                    cost = 0
                    for ci2, p2 in var_clauses[v]:
                        if p2 == ass[v] and sat[ci2] == 1:
                            cost += 1
                    # benefit weight
                    if mode == 0:
                        score = (cost, -weights[v])
                    elif mode == 1:
                        # prefer phase: lower cost, then prefer polarity that matches majority
                        pos = sum(1 for ci2, p2 in var_clauses[v] if p2)
                        phase_match = 1 if (p == (pos >= len(var_clauses[v]) - pos)) else 0
                        score = (cost, -phase_match, -weights[v])
                    else:
                        score = (cost, -weights[v])
                    if best_score is None or score < best_score:
                        best_score = score
                        best_v = v
                v = best_v
            flip_var(ass, v, sat)

            u = 0
            # quick check via sat
            # Instead of full recount, track if all >0
            all_sat = True
            for s in sat:
                if s == 0:
                    all_sat = False
                    break
            if all_sat:
                return ass

        track(ass)

    # Phase 2: if time remains, do targeted deep walks from best_assignment
    if best_assignment is not None and time.time() < deadline - 5:
        ass = best_assignment[:]
        sat = build_state(ass)
        u = count_unsat(ass)
        if u > 0:
            rng2 = random.Random(987654321)
            noise = 0.15
            while time.time() < deadline - 2:
                # find unsat clauses
                unsat_list = [ci for ci in range(m) if sat[ci] == 0]
                if not unsat_list:
                    return ass
                # pick a random unsat clause, but bias to shorter ones (easier)
                ci = unsat_list[rng2.randrange(len(unsat_list))]
                clause = clauses_opt[ci]
                if rng2.random() < noise:
                    v, p = clause[rng2.randrange(len(clause))]
                else:
                    best_v = clause[0][0]
                    best_cost = float('inf')
                    for v, p in clause:
                        cost = 0
                        for ci2, p2 in var_clauses[v]:
                            if p2 == ass[v] and sat[ci2] == 1:
                                cost += 1
                        if cost < best_cost:
                            best_cost = cost
                            best_v = v
                    v = best_v
                flip_var(ass, v, sat)
                # quick full check
                all_sat = True
                for s in sat:
                    if s == 0:
                        all_sat = False
                        break
                if all_sat:
                    return ass
                u2 = sum(1 for s in sat if s == 0)
                if u2 < best_unsat:
                    best_unsat = u2
                    best_assignment = ass[:]

    if best_assignment is not None:
        return best_assignment
    return ass
# EVOLVE-BLOCK-END
