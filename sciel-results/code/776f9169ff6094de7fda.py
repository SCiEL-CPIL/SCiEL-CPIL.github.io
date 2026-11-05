import time, random
from math import gcd

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N = 256
    # candidates 2..256
    cand = list(range(2, N+1))
    idx = {x:i for i,x in enumerate(cand)}
    m = len(cand)

    # conflict[i][j] = True if lcm(cand[i],cand[j]) <= 256
    # lcm = a*b//gcd
    conflict = [[False]*m for _ in range(m)]
    for i in range(m):
        a = cand[i]
        for j in range(i+1, m):
            b = cand[j]
            l = a*b//gcd(a,b)
            if l <= 256:
                conflict[i][j] = conflict[j][i] = True

    # weight = 1/a
    weight = [1.0/x for x in cand]

    # quick exact rational sum for a set of indices
    def rat_sum(idxs):
        from fractions import Fraction
        s = Fraction(0,1)
        for i in idxs:
            s += Fraction(1, cand[i])
        return s

    def score(idxs):
        return sum(weight[i] for i in idxs)

    def is_valid(idxs_set):
        # idxs_set is set of indices
        for i in idxs_set:
            for j in idxs_set:
                if i<j and conflict[i][j]:
                    return False
        return True

    def greedy(order):
        sel = set()
        for i in order:
            ok = True
            for j in sel:
                if conflict[i][j]:
                    ok = False
                    break
            if ok:
                sel.add(i)
        return sel

    best = None
    best_val = -1.0

    def consider(sel):
        nonlocal best, best_val
        if not sel:
            return
        v = score(sel)
        if v > best_val:
            best_val = v
            best = set(sel)

    # --- multiple randomized greedy starts ---
    rng = random.Random(12345)
    # orderings: ascending, descending, random, by weight desc, random weighted
    base_orders = [
        list(range(m)),
        list(range(m-1,-1,-1)),
        sorted(range(m), key=lambda i: -weight[i]),
        sorted(range(m), key=lambda i: weight[i]),
    ]
    for o in base_orders:
        sel = greedy(o)
        consider(sel)
        if time.time() > DEADLINE:
            break

    # local search polish
    def local_search(sel, max_iter=20000):
        sel = set(sel)
        cur = score(sel)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            if time.time() > DEADLINE:
                break
            # try add
            for i in range(m):
                if i in sel:
                    continue
                ok = True
                for j in sel:
                    if conflict[i][j]:
                        ok = False
                        break
                if ok:
                    ns = sel | {i}
                    nv = cur + weight[i]
                    if nv > cur + 1e-15:
                        sel = ns
                        cur = nv
                        improved = True
                        break
            if improved:
                continue
            # try swap: remove one, add one (net positive)
            for r in list(sel):
                # find best i not in sel, compatible with sel\{r}
                base = sel - {r}
                best_i = None
                best_gain = 0.0
                for i in range(m):
                    if i in sel:
                        continue
                    ok = True
                    for j in base:
                        if conflict[i][j]:
                            ok = False
                            break
                    if ok:
                        gain = weight[i] - weight[r]
                        if gain > best_gain + 1e-15:
                            best_gain = gain
                            best_i = i
                if best_i is not None:
                    ns = base | {best_i}
                    nv = cur + best_gain
                    if nv > cur + 1e-15:
                        sel = ns
                        cur = nv
                        improved = True
                        break
            if improved:
                continue
            # try 2-for-1: remove one, add two
            for r in list(sel):
                base = sel - {r}
                # find two compatible i,k not in sel, compatible with base and each other
                found = False
                # candidate additions compatible with base
                cands_add = [i for i in range(m) if i not in sel and all(not conflict[i][j] for j in base)]
                for ai in range(len(cands_add)):
                    i = cands_add[ai]
                    for bi in range(ai+1, len(cands_add)):
                        k = cands_add[bi]
                        if conflict[i][k]:
                            continue
                        gain = weight[i] + weight[k] - weight[r]
                        if gain > 1e-15:
                            sel = base | {i, k}
                            cur += gain
                            found = True
                            break
                    if found:
                        break
                if found:
                    improved = True
                    break
        return sel, cur

    # polish the best
    if best is not None:
        if time.time() < DEADLINE:
            sel, v = local_search(best)
            consider(sel)

    # randomized restarts with local search
    restarts = 0
    while time.time() < DEADLINE:
        restarts += 1
        # random ordering biased toward small numbers (high weight) but with noise
        order = list(range(m))
        # shuffle with bias: sort by (weight + noise)
        order = sorted(range(m), key=lambda i: -(weight[i] + rng.random()*0.02))
        sel = greedy(order)
        if time.time() < DEADLINE:
            sel, v = local_search(sel, max_iter=3000)
            consider(sel)
        if restarts > 500:
            break

    # final exact rational computation for reporting (not needed, evaluator computes)
    if best is None:
        # fallback
        best = {0}  # cand[0]=2
    return [cand[i] for i in best]
