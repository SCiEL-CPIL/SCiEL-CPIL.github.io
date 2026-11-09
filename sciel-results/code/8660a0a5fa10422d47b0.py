import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 295.0

    # Preprocess
    clean = []
    for c in clauses:
        seen = {}
        for l in c:
            seen[abs(l)] = l
        c2 = list(seen.values())
        pos = set(l for l in c2 if l > 0)
        neg = set(-l for l in c2 if l < 0)
        if pos & neg:
            continue
        if len(c2) == 0:
            return None
        clean.append(c2)
    clauses = clean
    m = len(clauses)

    # Static variable ordering by frequency (higher first), for tie-breaking
    freq = [0] * (n + 1)
    for c in clauses:
        for l in c:
            freq[abs(l)] += 1
    order = sorted(range(1, n + 1), key=lambda v: -freq[v])

    # assign: 0 unassigned, 1 true, -1 false (1-indexed)
    assign = [0] * (n + 1)
    trail = []

    def clause_status(c):
        """Return 2 if satisfied, 1 if unit (one unassigned), 0 if all unassigned, -1 if falsified."""
        sat = 0
        unassigned = 0
        for l in c:
            v = abs(l)
            if assign[v] == 0:
                unassigned += 1
            elif (l > 0 and assign[v] == 1) or (l < 0 and assign[v] == -1):
                sat += 1
                break
        if sat:
            return 2
        if unassigned == 0:
            return -1  # falsified
        if unassigned == 1:
            return 1
        return 0

    def unit_propagate():
        """Apply unit propagation. Returns False on conflict, True otherwise."""
        changed = True
        while changed:
            changed = False
            for c in clauses:
                st = clause_status(c)
                if st == -1:
                    return False
                if st == 1:
                    # find the unassigned literal
                    for l in c:
                        v = abs(l)
                        if assign[v] == 0:
                            assign[v] = 1 if l > 0 else -1
                            trail.append(v)
                            changed = True
                            break
        return True

    def all_satisfied():
        for c in clauses:
            if clause_status(c) != 2:
                return False
        return True

    def pick_var():
        # pick unassigned var with highest frequency
        for v in order:
            if assign[v] == 0:
                return v
        return None

    def dpll(depth):
        if time.time() > deadline - 5:
            return None
        if not unit_propagate():
            return None
        if all_satisfied():
            return assign[1:]
        v = pick_var()
        if v is None:
            return assign[1:] if all_satisfied() else None
        # try both values
        for val in (1, -1):
            assign[v] = val
            trail.append(v)
            res = dpll(depth + 1)
            if res is not None:
                return res
            # undo
            while trail and trail[-1] == v:
                trail.pop()
            assign[v] = 0
        return None

    import sys
    sys.setrecursionlimit(100000)
    result = dpll(0)
    if result is not None:
        ok = all(any(((l > 0 and result[abs(l) - 1]) or (l < 0 and not result[abs(l) - 1])) for l in c) for c in clauses)
        if ok:
            return result

    # Fallback WalkSAT
    rng = random.Random(42)
    deadline = start_time + 290.0
    best, best_uc = None, float('inf')
    while time.time() < deadline:
        a = [rng.random() < 0.5 for _ in range(n)]
        uc = sum(1 for c in clauses if not any(((l > 0 and a[abs(l) - 1]) or (l < 0 and not a[abs(l) - 1])) for l in c))
        if uc == 0:
            return a
        if uc < best_uc:
            best_uc, best = uc, a[:]
        for _ in range(30000):
            if time.time() > deadline:
                break
            if uc == 0:
                break
            unsat = [i for i, c in enumerate(clauses) if not any(((l > 0 and a[abs(l) - 1]) or (l < 0 and not a[abs(l) - 1])) for l in c)]
            ci = rng.choice(unsat)
            l = rng.choice(clauses[ci])
            v = abs(l) - 1
            a[v] = not a[v]
            new_uc = sum(1 for cc in clauses if not any(((ll > 0 and a[abs(ll) - 1]) or (ll < 0 and not a[abs(ll) - 1])) for ll in cc))
            if new_uc >= uc:
                a[v] = not a[v]
            else:
                uc = new_uc
                if uc < best_uc:
                    best_uc, best = uc, a[:]
        if best_uc == 0:
            return best
    return best if best is not None else [False] * n
