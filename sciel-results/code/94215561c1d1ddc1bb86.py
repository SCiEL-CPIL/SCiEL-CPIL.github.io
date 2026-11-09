def construct():
    import time
    t0 = time.time()
    DEADLINE = 78.0  # safety margin before the 90 s wall-clock kill

    from math import gcd
    import numpy as np

    N = 256
    nums = list(range(2, N + 1))
    n = len(nums)
    recip = np.array([1.0 / x for x in nums])

    # conflict graph: i ~ j  iff  lcm(nums[i], nums[j]) <= N
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            g = gcd(a, b)
            if a * b // g <= N:
                conflicts[i].add(j)
                conflicts[j].add(i)

    def valid(S):
        if not S:
            return False
        lst = list(S)
        for ii in range(len(lst)):
            for jj in range(ii + 1, len(lst)):
                if lst[jj] in conflicts[lst[ii]]:
                    return False
        return True

    def value(S):
        return float(sum(recip[i] for i in S))

    # seed (same greedy as the reference program)
    seed = set()
    for x in range(2, N + 1):
        i = x - 2
        ok = True
        for u in seed:
            if u in conflicts[i]:
                ok = False
                break
        if ok:
            seed.add(i)

    best = set(seed)
    best_val = value(best)

    def local_search(S):
        S = set(S)
        inS = [False] * n
        for i in S:
            inS[i] = True
        improved = True
        it = 0
        while improved and it < 400:
            improved = False
            it += 1
            # move: add v, displacing all its current neighbours C(v)
            for v in range(n):
                if inS[v]:
                    continue
                C = [u for u in conflicts[v] if inS[u]]
                cost = float(sum(recip[u] for u in C))
                if recip[v] > cost + 1e-12:
                    for u in C:
                        inS[u] = False
                        S.discard(u)
                    inS[v] = True
                    S.add(v)
                    improved = True
        return S, value(S)

    # local search from the seed
    ls, lsv = local_search(best)
    if lsv > best_val:
        best, best_val = ls, lsv

    # ILP (HiGHS) -- weighted independent set
    if time.time() - t0 < DEADLINE - 22:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import scipy.sparse as sp
            rows = []
            cols = []
            data = []
            for i in range(n):
                for j in conflicts[i]:
                    if i < j:
                        rows.append(len(rows)); cols.append(i); data.append(1.0)
                        rows.append(len(rows)); cols.append(j); data.append(1.0)
            m = len(rows)
            A = sp.csr_matrix((data, (rows, cols)), shape=(m, n))
            c = -recip  # minimize -sum(1/a)  ==  maximize sum(1/a)
            cons = LinearConstraint(A, 0, 1)
            integ = np.ones(n)
            bnds = Bounds(0, 1)
            tl = max(5.0, min(36.0, DEADLINE - (time.time() - t0) - 16.0))
            res = milp(c, constraints=cons, integrality=integ, bounds=bnds,
                       options={'time_limit': tl, 'mip_rel_gap': 1e-4})
            if res.x is not None:
                sol = set(int(i) for i in np.where(res.x > 0.5)[0])
                if valid(sol):
                    v = value(sol)
                    if v > best_val:
                        best, best_val = sol, v
                base = sol if valid(sol) else best
                pol, polv = local_search(base)
                if polv > best_val:
                    best, best_val = pol, polv
        except Exception:
            pass

    # random perturbations to escape local optima
    if time.time() - t0 < DEADLINE - 16:
        import random
        rng = random.Random(12345)
        for _ in range(30):
            if time.time() - t0 > DEADLINE - 14:
                break
            cand = set(best)
            for _ in range(rng.randint(1, 4)):
                if cand:
                    cand.discard(rng.choice(sorted(cand)))
            cand, cv = local_search(cand)
            if cv > best_val:
                best, best_val = cand, cv

    if not best or not valid(best):
        best = set(seed)

    return [nums[i] for i in sorted(best)]
