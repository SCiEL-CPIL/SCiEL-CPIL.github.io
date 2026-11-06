import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def is_valid(arr):
        if len(arr) > 32 or 0 not in arr or max(arr) > 10**6:
            return False
        diffs = set()
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return all(d in diffs for d in range(1, 31))

    def solve(D, max_time):
        best = None
        def bt(sel, used, cov, mn):
            nonlocal best
            if best is not None or time.time() > max_time:
                return
            if len(cov) == 30:
                best = sel[:]
                return
            rem = D - mn + 1
            if rem <= 0:
                return
            if len(sel) + rem > 32:
                rem = 32 - len(sel)
                if rem <= 0:
                    return
            if rem * len(sel) < 30 - len(cov):
                return
            cands = []
            for x in range(mn, D + 1):
                if x in sel:
                    continue
                ok = True
                nd = []
                for s in sel:
                    d = x - s
                    if d <= 0:
                        continue
                    if d in used:
                        ok = False
                        break
                    nd.append(d)
                if not ok:
                    continue
                nc = sum(1 for d in nd if 1 <= d <= 30 and d not in cov)
                cands.append((x, nc, nd))
            if not cands:
                return
            cands.sort(key=lambda t: (-t[1], t[0]))
            for x, nc, nd in cands:
                if best is not None or time.time() > max_time:
                    return
                bt(sel + [x], used | set(nd), cov | set(d for d in nd if 1 <= d <= 30), x + 1)
        bt([0], set(), set(), 1)
        return best

    for D in range(45, 201):
        if time.time() > deadline - 10:
            break
        r = solve(D, deadline - 5)
        if r is not None and is_valid(r):
            return r

    for D in range(200, 501):
        if time.time() > deadline - 3:
            break
        r = solve(D, deadline - 2)
        if r is not None and is_valid(r):
            return r

    # Greedy fallback
    def greedy():
        selected = [0]
        used = set()
        for target in range(1, 31):
            if target in used:
                continue
            best_x = None
            best_cover = -1
            for x in range(1, 5000):
                if x in selected:
                    continue
                conflict = False
                new_diffs = []
                for s in selected:
                    d = x - s
                    if d <= 0:
                        continue
                    if d in used:
                        conflict = True
                        break
                    new_diffs.append(d)
                if conflict:
                    continue
                cover = sum(1 for d in new_diffs if 1 <= d <= 30 and d not in used)
                if target in new_diffs:
                    best_x = x
                    break
                elif cover > best_cover:
                    best_x = x
                    best_cover = cover
            if best_x is not None:
                selected.append(best_x)
                for s in selected[:-1]:
                    d = best_x - s
                    if d > 0:
                        used.add(d)
        if is_valid(selected):
            return sorted(selected)
        return None

    result = greedy()
    if result is not None:
        return result

    r = solve(1000, deadline - 1)
    if r is not None and is_valid(r):
        return r

    return [0, 1, 4, 10, 12, 28, 32, 45, 63]

# EVOLVE-BLOCK-END
