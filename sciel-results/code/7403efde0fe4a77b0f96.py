import time

def construct():
    deadline = time.time() + 140.0
    NEED = frozenset(range(1, 31))
    NEED_SET = set(range(1, 31))

    def is_sidon_and_cover(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False, None
                diffs.add(d)
        if NEED_SET.issubset(diffs):
            return True, diffs
        return False, diffs

    # Greedy initial
    def greedy():
        S = [0]
        diffs = set()
        for d in range(1, 31):
            if d in diffs:
                continue
            placed = False
            for s in sorted(S):
                for x in (s + d, s - d):
                    if x < 0 or x in S:
                        continue
                    # check sidon
                    ok = True
                    nd = set(diffs)
                    for t in S:
                        dd = x - t if x > t else t - x
                        if dd in nd:
                            ok = False
                            break
                        nd.add(dd)
                    if ok:
                        S = sorted(S + [x])
                        diffs = nd
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                y = 2 * max(S) + 3 * d + 1
                while True:
                    ok = True
                    nd = set(diffs)
                    for t in S:
                        dd = y - t
                        if dd in nd:
                            ok = False
                            break
                        nd.add(dd)
                    if ok:
                        S = sorted(S + [y])
                        diffs = nd
                        break
                    y += 1
        return S

    best = greedy()
    best_max = max(best)

    # DFS to improve
    best_sol = list(best)
    best_max_val = best_max

    def dfs(S, cov, last):
        nonlocal best_max_val, best_sol
        if time.time() > deadline:
            return
        if NEED_SET.issubset(cov):
            if last < best_max_val:
                best_max_val = last
                best_sol = list(S)
            return
        if last >= best_max_val:
            return
        if len(S) >= 32:
            return
        # candidates: x = s + d for s in S, d in NEED, x > last, x < best_max_val
        cands = set()
        missing = NEED_SET - cov
        for s in S:
            for d in missing:
                x = s + d
                if last < x < best_max_val:
                    cands.add(x)
        for x in sorted(cands):
            # Sidon check for new diffs x - s
            ok = True
            nd = set(cov)
            for s in S:
                dd = x - s
                if dd in nd:
                    ok = False
                    break
                nd.add(dd)
            if not ok:
                continue
            dfs(S + [x], nd, x)

    dfs([0], set(), 0)

    return best_sol
