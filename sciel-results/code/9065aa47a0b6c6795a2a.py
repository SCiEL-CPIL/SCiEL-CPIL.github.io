def construct():
    import time
    t0 = time.time()
    DEADLINE = t0 + 110.0  # safety margin under the 150s kill

    def diffs_of(s):
        out = []
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n):
                out.append(s[j] - s[i])
        return out

    # Greedy baseline (valid solution) to seed an upper bound
    def greedy():
        a = [0]
        for d in range(1, 31):
            if d in diffs_of(a):
                continue
            placed = False
            cands = sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a))
            for x in cands:
                trial = sorted(a + [x])
                if len(set(diffs_of(trial))) == len(diffs_of(trial)):
                    a = trial
                    placed = True
                    break
            if not placed:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs_of(trial))) == len(diffs_of(trial)):
                        a = trial
                        break
                    y += 1
        return a

    best = greedy()
    best_max = max(best)

    import sys
    sys.setrecursionlimit(100000)

    def dfs(a, aset, diff_set):
        nonlocal best, best_max
        if time.time() > DEADLINE:
            return
        n = len(a)
        # Completion check first
        if all(d in diff_set for d in range(1, 31)):
            if a[-1] < best_max:
                best_max = a[-1]
                best = list(a)
            return
        if n >= 32:
            return
        if a[-1] >= best_max:
            return
        missing_set = set(d for d in range(1, 31) if d not in diff_set)
        m = len(missing_set)
        r = 32 - n
        # Prune: not enough new differences possible to cover all missing
        if m > r * n + r * (r - 1) // 2:
            return
        # Forbidden x: x = ai +/- d (d existing diff)  OR  x midpoint of two elements
        forbidden = set()
        for ai in a:
            for d in diff_set:
                forbidden.add(ai + d)
                forbidden.add(ai - d)
        for i in range(n):
            for j in range(i + 1, n):
                s = a[i] + a[j]
                if s % 2 == 0:
                    forbidden.add(s // 2)
        cands = []
        for x in range(1, best_max):
            if x in aset or x in forbidden:
                continue
            newd = [abs(x - ai) for ai in a]
            cnt = 0
            for d in newd:
                if d in missing_set:
                    cnt += 1
            cands.append((-cnt, x, newd))
        cands.sort()
        for negcnt, x, newd in cands:
            if time.time() > DEADLINE:
                return
            new_a = list(a)
            new_a.append(x)
            new_a.sort()
            dfs(new_a, aset | {x}, diff_set | set(newd))

    try:
        dfs([0], {0}, set())
        # A few diverse roots to escape local structure
        for start in ([0, 1], [0, 1, 3], [0, 2], [0, 3]):
            if time.time() > DEADLINE:
                break
            sa = sorted(start)
            aset = set(sa)
            ds_list = diffs_of(sa)
            if len(set(ds_list)) != len(ds_list):
                continue
            dfs(sa, aset, set(ds_list))
    except Exception:
        pass

    return best
