"""Hybrid DPLL + vectorized WalkSAT for planted 3-SAT (150 vars)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _luby(k):
    m = 1
    while m < k:
        m *= 2
    while m != 1:
        if k == m:
            return k
        k -= m
        m //= 2
    return 1


def _dpll(n, clauses, rng, deadline):
    """DPLL with unit propagation, pure literals, fail-first.
    Returns list of bools (satisfying) or None."""
    # assign: 0 unknown, 1 true, -1 false
    assign = [0] * n

    def propagate(clauses):
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for c in clauses:
                seen = 0
                unit = None
                for l in c:
                    v = assign[abs(l) - 1]
                    if v == 0:
                        seen += 1
                        unit = l
                    elif v == 1:
                        if l > 0:
                            seen = 0
                            break
                    else:
                        if l < 0:
                            seen = 0
                            break
                if seen == 0:
                    continue
                elif seen == 1:
                    l = unit
                    v = abs(l) - 1
                    val = 1 if l > 0 else -1
                    if assign[v] == 0:
                        assign[v] = val
                        changed = True
                    elif assign[v] != val:
                        return None
                    continue
                else:
                    new_clauses.append(c)
            clauses = new_clauses
        return clauses

    def pure_literal(clauses):
        pos = set()
        neg = set()
        for c in clauses:
            for l in c:
                if l > 0:
                    pos.add(l)
                else:
                    neg.add(l)
        pure = (pos - neg) | (neg - pos)
        if not pure:
            return
        for l in pure:
            v = abs(l) - 1
            if assign[v] == 0:
                assign[v] = 1 if l > 0 else -1

    def choose_var(clauses):
        best_c = None
        best_len = 999
        for c in clauses:
            cnt = sum(1 for l in c if assign[abs(l) - 1] == 0)
            if cnt < best_len:
                best_len = cnt
                best_c = c
                if cnt == 1:
                    break
        if best_c is None:
            return -1
        cands = [l for l in best_c if assign[abs(l) - 1] == 0]
        if not cands:
            return -1
        l = rng.choice(cands)
        return abs(l) - 1

    def search(clauses):
        if time.time() > deadline:
            return None
        clauses = propagate(clauses)
        if clauses is None:
            return False
        if not clauses:
            return True
        pure_literal(clauses)
        clauses = propagate(clauses)
        if clauses is None:
            return False
        if not clauses:
            return True
        v = choose_var(clauses)
        if v == -1:
            return False
        for val in (1, -1):
            if time.time() > deadline:
                return None
            backup = assign[:]
            assign[v] = val
            res = search(clauses)
            if res is True:
                return True
            if res is None:
                assign[:] = backup
                return None
            assign[:] = backup
        return False

    clauses0 = [list(c) for c in clauses]
    res = search(clauses0)
    if res is True:
        return [bool(a) for a in assign]
    return None


def _walksat(n, clauses, start, rng, deadline, max_steps=200000):
    m = len(clauses)
    arr = np.array(clauses, dtype=np.int64)
    var_idx = np.abs(arr) - 1
    signs = arr > 0

    if start is not None:
        a = np.array([bool(x) for x in start], dtype=bool)
    else:
        a = np.array(rng.random(n) < 0.5, dtype=bool)

    def is_sat(a):
        return (a[var_idx] == signs).any(axis=1)

    sat = is_sat(a)
    if sat.all():
        return a.tolist()

    var_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for li in range(3):
            var_clauses[int(var_idx[ci, li])].append((ci, bool(signs[ci, li])))

    cur_unsat = int((~sat).sum())

    for step in range(max_steps):
        if time.time() > deadline:
            return a.tolist()
        if cur_unsat == 0:
            return a.tolist()
        unsat_idx = np.where(~sat)[0]
        if len(unsat_idx) == 0:
            return a.tolist()
        ci = int(rng.choice(unsat_idx))
        lits = [(int(var_idx[ci, li]), bool(signs[ci, li])) for li in range(3)]
        rng.shuffle(lits)
        best_gain = -1
        best_v = None
        old_sat = sat
        for v, s in lits:
            new_v = not a[v]
            delta = 0
            for ci2, s2 in var_clauses[v]:
                was_sat = old_sat[ci2]
                other_sat = False
                for li in range(3):
                    if var_idx[ci2, li] == v:
                        continue
                    if a[var_idx[ci2, li]] == signs[ci2, li]:
                        other_sat = True
                        break
                v_sat_new = (new_v == s2)
                is_sat_new = other_sat or v_sat_new
                if was_sat and not is_sat_new:
                    delta += 1
                elif (not was_sat) and is_sat_new:
                    delta -= 1
            gain = -delta
            if gain > best_gain:
                best_gain = gain
                best_v = v
        a[best_v] = not a[best_v]
        sat = is_sat(a)
        cur_unsat = int((~sat).sum())

    return a.tolist()


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 280
    rng = random.Random(12345)

    # Phase 1: DPLL with restarts
    restart = 0
    while time.time() < deadline - 30:
        budget = min(15.0, 0.5 * _luby(restart + 1))
        sub_deadline = min(deadline - 30, time.time() + budget)
        assign = _dpll(n, clauses, rng, sub_deadline)
        if assign is not None:
            if all(any(assign[abs(l) - 1] == (l > 0) for l in c) for c in clauses):
                return assign
        restart += 1
        if restart > 300:
            break

    # Phase 2: WalkSAT with restarts
    start = None
    while time.time() < deadline - 5:
        ws_deadline = min(deadline - 2, time.time() + 25)
        res = _walksat(n, clauses, start, rng, ws_deadline, max_steps=400000)
        if res is not None and all(any(res[abs(l) - 1] == (l > 0) for l in c) for c in clauses):
            return res
        start = res

    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
