"""Rewritten program: base + greedy + local-search for the no-forbidden-triple set."""
import time
import random

N = 128


def _valid(s):
    """Return True iff set s has no forbidden triple."""
    m = set(s)
    for a in s:
        for b in s:
            if a >= b:
                continue
            for c in s:
                if b >= c:
                    continue
                if (a + b) in m and (a + c) in m and (b + c) in m:
                    return False
    return True


def _greedy_add(cand, s, pairs, sums, sums_by, n):
    """Try to add cand; return (ok, new_s, new_pairs, new_sums, new_sums_by)."""
    ok = True
    for (x, y) in pairs:
        if x == cand or y == cand:
            continue
        if (x + y) in sums:
            ok = False
            break
    if not ok:
        return (False, s, pairs, sums, sums_by)
    ns = set(s)
    ns.add(cand)
    npairs = list(pairs)
    nsums = set(sums)
    nsums_by = {k: list(v) for k, v in sums_by.items()}
    for x in s:
        if x == cand:
            continue
        npairs.append((x, cand))
        s1 = x + cand
        if s1 <= n:
            nsums.add(s1)
            nsums_by.setdefault(s1, []).append(cand)
    return (True, ns, npairs, nsums, nsums_by)


def _local_search(s, n, budget):
    """Try to grow s via add-one and add-one/drop-one-or-two moves."""
    m = set(s)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < budget:
        improved = False
        # order candidates by fewest new sums (less likely to conflict)
        cands = [x for x in range(1, n + 1) if x not in m]
        cands.sort(key=lambda x: sum(1 for y in m if x + y <= n))
        for cand in cands:
            if (time.time() - start) > budget:
                break
            ok = True
            for a in m:
                for b in m:
                    if a >= b:
                        continue
                    if a == cand or b == cand:
                        continue
                    if (a + b) in m:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                m.add(cand)
                improved = True
                break
        # try add cand, drop d1, (optionally d2)
        cands2 = [x for x in range(1, n + 1) if x not in m]
        cands2.sort(key=lambda x: sum(1 for y in m if x + y <= n))
        for cand in cands2:
            if (time.time() - start) > budget:
                break
            # find minimal drop set
            for d1 in list(m):
                if d1 == cand:
                    continue
                trial = m - {d1}
                trial.add(cand)
                if _valid(trial):
                    m = trial
                    improved = True
                    break
            if improved or (time.time() - start) > budget:
                break
        if not improved:
            # try two-drop moves (only if still under budget)
            if (time.time() - start) < budget:
                cands3 = [x for x in range(1, n + 1) if x not in m]
                cands3.sort(key=lambda x: sum(1 for y in m if x + y <= n))
                ml = list(m)
                for cand in cands3:
                    if (time.time() - start) > budget:
                        break
                    for i in range(len(ml)):
                        d1 = ml[i]
                        if d1 == cand:
                            continue
                        for j in range(i + 1, len(ml)):
                            d2 = ml[j]
                            if d2 == cand:
                                continue
                            trial = m - {d1, d2}
                            trial.add(cand)
                            if _valid(trial):
                                m = trial
                                improved = True
                                break
                        if improved or (time.time() - start) > budget:
                            break
                    if improved or (time.time() - start) > budget:
                        break
    return m


def construct():
    deadline = time.time() + 75.0  # safety margin under 90s
    rng = random.Random(12345)

    # Base: upper half (64 elements) is always valid.
    base = set(range(65, 129))

    # Greedy add lower-half elements that don't create a forbidden triple.
    pairs = [(a, b) for a in base for b in base if a < b]
    sums = set()
    sums_by = {}
    for (x, y) in pairs:
        s = x + y
        if s <= N:
            sums.add(s)
            sums_by.setdefault(s, []).append(x)
    s = base
    for cand in range(1, 65):
        ok, s, pairs, sums, sums_by = _greedy_add(cand, s, pairs, sums, sums_by, N)
        if (time.time() > deadline):
            break

    best = set(s)

    # Local search polish (add / add-drop / add-2drop), with restarts.
    ls_budget = min(20.0, max(2.0, deadline - time.time()))
    s2 = _local_search(set(s), N, ls_budget)
    if len(s2) > len(best):
        best = s2

    # A couple of randomized restarts if time remains, keeping the best.
    while time.time() < deadline - 5.0:
        start = time.time()
        s3 = set(base)
        order = list(range(1, 65))
        rng.shuffle(order)
        pairs3 = [(a, b) for a in s3 for b in s3 if a < b]
        sums3 = set()
        sums_by3 = {}
        for (x, y) in pairs3:
            sv = x + y
            if sv <= N:
                sums3.add(sv)
                sums_by3.setdefault(sv, []).append(x)
        for cand in order:
            if time.time() > deadline - 5.0:
                break
            ok, s3, pairs3, sums3, sums_by3 = _greedy_add(cand, s3, pairs3, sums3, sums_by3, N)
        ls_b = min(15.0, max(1.0, deadline - time.time() - 5.0))
        s4 = _local_search(s3, N, ls_b)
        if len(s4) > len(best):
            best = s4
        if time.time() > deadline - 5.0:
            break

    res = sorted(int(x) for x in best)
    return res
