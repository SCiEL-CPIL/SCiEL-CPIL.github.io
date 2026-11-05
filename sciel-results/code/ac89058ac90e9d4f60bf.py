def construct():
    import time
    from math import gcd, lcm

    start = time.time()
    deadline = start + 75.0

    numbers = list(range(2, 257))
    n = len(numbers)

    # Precompute conflict matrix
    conflict = [[False]*n for _ in range(n)]
    for i in range(n):
        a = numbers[i]
        for j in range(i+1, n):
            b = numbers[j]
            if lcm(a, b) <= 256:
                conflict[i][j] = True
                conflict[j][i] = True

    def is_valid(sel_set):
        lst = list(sel_set)
        for p in range(len(lst)):
            for q in range(p+1, len(lst)):
                if conflict[numbers.index(lst[p])][numbers.index(lst[q])]:
                    return False
        return True

    best = None
    best_val = 0.0

    def val(sel):
        return sum(1.0/x for x in sel)

    # Try MIP
    try:
        import numpy as np
        from scipy.optimize import LinearConstraint, milp, Bounds
        from scipy.sparse import coo_matrix

        c = np.array([-1.0/numbers[i] for i in range(n)])
        rows, cols, vals = [], [], []
        for i in range(n):
            for j in range(i+1, n):
                if conflict[i][j]:
                    rows.append(len(rows)); cols.append(i); vals.append(1.0)
                    rows.append(len(rows)); cols.append(j); vals.append(1.0)
        A = coo_matrix((vals, (rows, cols)), shape=(len(rows), n)).tocsr()
        constraints = [LinearConstraint(A, -np.inf, 1)]
        bounds = Bounds(0, 1)
        integrality = np.ones(n)
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0:
            sel = [numbers[i] for i in range(n) if res.x[i] > 0.5]
            if sel and val(sel) > best_val:
                best = sel
                best_val = val(sel)
    except Exception:
        pass

    # Local search improvement
    if best is not None:
        sel = set(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding
            for i in range(n):
                if time.time() > deadline: break
                if i not in sel and all(not conflict[i][j] for j in sel):
                    sel.add(i); improved = True
            # Try swaps
            for i in range(n):
                if time.time() > deadline: break
                if i in sel: continue
                for j in list(sel):
                    if all(not conflict[i][k] for k in sel - {j}):
                        if 1.0/numbers[i] > 1.0/numbers[j]:
                            sel.discard(j); sel.add(i); improved = True
                            break
        if val(list(sel)) > best_val:
            best = list(sel); best_val = val(list(sel))
    else:
        # Fallback greedy
        sel = []
        order = sorted(range(n), key=lambda i: -1.0/numbers[i])
        for i in order:
            if time.time() > deadline: break
            if all(not conflict[i][j] for j in sel):
                sel.append(i)
        best = [numbers[i] for i in sel]

    return best
