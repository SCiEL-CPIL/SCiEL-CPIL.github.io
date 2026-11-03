import time
import random

def _validate(rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ri
                if d in used:
                    return False
                used.add(d)
    return True

def _greedy(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _backtrack(m, n_rows, k, deadline, order_mode=0, seed=0):
    """Return a valid solution with max entry <= m, or None."""
    rows = [[0] for _ in range(n_rows)]
    order = [(r, lvl) for lvl in range(1, k + 1) for r in range(n_rows)]
    found = [None]
    rng = random.Random(seed)
    
    def search(idx, used):
        if time.time() > deadline:
            return False
        if idx == len(order):
            found[0] = [r[:] for r in rows]
            return True
        r, lvl = order[idx]
        lo = rows[r][lvl - 1] + 1
        if lvl == 1 and r > 0:
            lo = max(lo, rows[r - 1][1])
        row = rows[r]
        # build candidate x list
        if order_mode == 0:
            xs = range(lo, m + 1)
        elif order_mode == 1:
            xs = range(m, lo - 1, -1)
        else:
            xs = list(range(lo, m + 1))
            rng.shuffle(xs)
        for x in xs:
            nd = 0
            for a in row:
                nd |= 1 << (x - a)
            if nd & used:
                continue
            row.append(x)
            if search(idx + 1, used | nd):
                return True
            row.pop()
        return False
    
    search(0, 0)
    return found[0]

def _local_search(n_rows, k, initial, deadline):
    def energy(state):
        dc = {}
        max_e = 0
        for row in state:
            for v in row:
                if v > max_e:
                    max_e = v
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    dc[d] = dc.get(d, 0) + 1
        coll = sum(c - 1 for c in dc.values() if c > 1)
        return coll * 100000 + max_e
    
    state = [row[:] for row in initial]
    best_state = [row[:] for row in state]
    best_e = energy(state)
    cur_e = best_e
    T = 100.0
    T_min = 0.5
    cool = 0.9995
    rng = random.Random(123)
    U = max(max(r) for r in initial) + 20  # upper bound for values
    
    while time.time() < deadline and T > T_min:
        r = rng.randrange(n_rows)
        lvl = rng.randrange(1, k + 1)
        lo = state[r][lvl - 1] + 1
        if lvl < k:
            hi = state[r][lvl + 1] - 1
        else:
            hi = min(U, state[r][k] + 10)
        if hi < lo:
            hi = lo
        x = rng.randrange(lo, hi + 1)
        if x == state[r][lvl]:
            T *= cool
            continue
        old = state[r][lvl]
        state[r][lvl] = x
        new_e = energy(state)
        dE = new_e - cur_e
        if dE <= 0 or rng.random() < pow(2.718281828, -dE / T):
            cur_e = new_e
            if new_e < best_e:
                best_e = new_e
                best_state = [row[:] for row in state]
        else:
            state[r][lvl] = old
        T *= cool
    
    # return best_state only if valid
    if _validate(best_state, k):
        return best_state
    return None

def construct(n_rows, k):
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s
    
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs
    
    # Phase 1: greedy
    best_sol = _greedy(n_rows, k)
    best_m = max(max(r) for r in best_sol)
    
    # Phase 2: local search (if time)
    if time.time() < deadline - 60:
        ls = _local_search(n_rows, k, best_sol, deadline - 60)
        if ls is not None:
            best_sol = ls
            best_m = max(max(r) for r in best_sol)
    
    # Phase 3: backtracking over m
    remaining = deadline - time.time()
    if remaining > 3:
        upper = best_m
        # try m from lower_bound to upper - 1
        m_vals = list(range(lower_bound, upper))
        # if too many, cap
        if len(m_vals) > 30:
            m_vals = m_vals[:30]
        n_m = len(m_vals)
        for i, m in enumerate(m_vals):
            if time.time() > deadline - 2:
                break
            rem = deadline - 2 - time.time()
            per_m = rem / max(1, n_m - i)
            per_m = min(per_m, 15.0)
            # try a few order modes
            modes = [0, 1, 2]
            sub = per_m / len(modes)
            for mode in modes:
                if time.time() > deadline - 2:
                    break
                t_end = time.time() + sub
                sol = _backtrack(m, n_rows, k, t_end, order_mode=mode, seed=m * 31 + mode)
                if sol is not None and _validate(sol, k):
                    best_sol = sol
                    best_m = m
                    # optimal (smallest m found); return
                    return best_sol
    return best_sol
