import time, random

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 112.0
    row_len = k + 1

    def row_diffs(row):
        d = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d.add(row[j] - ri)
        return d

    def greedy_row(used):
        row = [0]; x = 0
        while len(row) < row_len:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x); used |= new
        return row

    def find_row(used, best_max_init, S_cap, node_budget, dl, rng):
        """Search a row_len-mark row (starts 0) with max < best_max_init,
        all its differences avoiding `used`. Returns (row, diffs) or None."""
        best = [None]; best_max = [best_max_init]; nodes = [0]
        def dfs(row, rdiffs, last):
            if time.time() > dl or nodes[0] >= node_budget:
                return
            nodes[0] += 1
            if len(row) == row_len:
                m = row[-1]
                if m < best_max[0]:
                    best_max[0] = m; best[0] = row[:]
                return
            hi = min(best_max[0] - 1, S_cap); lo = last + 1
            if lo > hi:
                return
            for x in range(lo, hi + 1):
                new = {x - a for a in row}
                if new & used or new & rdiffs:
                    continue
                row.append(x); rdiffs |= new
                dfs(row, rdiffs, x)
                rdiffs -= new; row.pop()
                if nodes[0] >= node_budget or time.time() > dl:
                    return
        dfs([0], set(), 0)
        if best[0] is None:
            return None
        return best[0], row_diffs(best[0])

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def run(seed, dl):
        rng = random.Random(seed)
        used = set(); rows = []
        for _ in range(n_rows):
            if time.time() > dl:
                break
            res = find_row(used, 200, 200, 200000, dl, rng)
            if res is None:
                row = greedy_row(used); used |= row_diffs(row)
            else:
                row, rd = res; used |= rd
            rows.append(row)
        while len(rows) < n_rows:
            row = greedy_row(used); used |= row_diffs(row); rows.append(row)

        best_rows = [r[:] for r in rows]; best_scope = scope_of(rows)

        for _ in range(600):
            if time.time() > dl:
                break
            improved = False
            order = list(range(n_rows))
            order.sort(key=lambda i: (-rows[i][-1], rng.random()))
            used = set()
            for r in rows:
                used |= row_diffs(r)
            for idx in order:
                if time.time() > dl:
                    break
                r = rows[idx]; d = row_diffs(r); used -= d
                cur = r[-1]
                res = find_row(used, cur, cur, 120000, dl, rng)
                if res is not None:
                    nr, nd = res; rows[idx] = nr; used |= nd; improved = True
                else:
                    used |= d
            if not improved:
                if time.time() > dl:
                    break
                pi = rng.randrange(n_rows)
                d = row_diffs(rows[pi]); used -= d
                rows[pi] = greedy_row(used); used |= row_diffs(rows[pi])
                continue
            s = scope_of(rows)
            if s < best_scope:
                best_scope = s; best_rows = [r[:] for r in rows]
        return best_scope, best_rows

    try:
        overall = None; seed = 0; first = True
        while time.time() < DEADLINE - 5:
            if first:
                dl = min(DEADLINE - 3, time.time() + 70); first = False
            else:
                dl = min(DEADLINE - 3, time.time() + 18)
            sc, rows = run(seed, dl)
            if overall is None or sc < overall[0]:
                overall = (sc, rows)
            seed += 1
        if overall is not None:
            return overall[1]
    except Exception:
        pass
    # last-resort valid greedy
    used = set(); rows = []
    for _ in range(n_rows):
        row = greedy_row(used); used |= row_diffs(row); rows.append(row)
    return rows
