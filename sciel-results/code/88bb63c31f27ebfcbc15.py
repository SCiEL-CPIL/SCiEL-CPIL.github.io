import time
import random
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin under 150s

    def _elapsed():
        return time.time() - start_time

    def _row_from_deltas(deltas):
        row = [0]
        s = 0
        for d in deltas:
            s += d
            row.append(s)
        return row

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _scope(rows):
        return max(max(r) for r in rows)

    def _valid(rows):
        all_d = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            rd = _row_diffs(row)
            if rd & all_d:
                return False
            all_d |= rd
        return True

    def _counting_lower_bound():
        return n_rows * k * (k + 1) // 2

    # ---------- Greedy (element space) for a quick feasible start ----------
    def greedy_construct(seed_offset=0):
        rng = random.Random(7 + seed_offset)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def local_search_shrink(rows, dl, rng):
        rows = [list(r) for r in rows]
        used = set()
        for r in rows:
            used |= _row_diffs(r)
        improved = True
        while improved and _elapsed() < dl:
            improved = False
            cur_scope = _scope(rows)
            for ri in range(len(rows)):
                if _elapsed() > dl:
                    break
                row = rows[ri]
                for pos in range(len(row) - 1, 0, -1):
                    if _elapsed() > dl:
                        break
                    old_val = row[pos]
                    if old_val < cur_scope:
                        continue
                    low = row[pos - 1] + 1
                    high = old_val - 1
                    if pos + 1 < len(row):
                        high = min(high, row[pos + 1] - 1)
                    if low > high:
                        continue
                    cands = list(range(high, low - 1, -1))
                    rng.shuffle(cands)
                    for new_val in cands:
                        if _elapsed() > dl:
                            break
                        if new_val >= old_val:
                            continue
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(old_val - row[i])
                        for i in range(pos + 1, len(row)):
                            old_diffs.add(row[i] - old_val)
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for i in range(pos + 1, len(row)):
                            new_diffs.add(row[i] - new_val)
                        if new_diffs & (used - old_diffs):
                            continue
                        row[pos] = new_val
                        used -= old_diffs
                        used |= new_diffs
                        improved = True
                        break
        return rows

    # ---------- Exact cover over difference cells (DLX / Algorithm X) ----------
    # Represent each candidate row by (deltas tuple, diffset frozenset, row list).
    # Columns = difference cells 1..m. A solution picks n_rows candidates whose
    # diffsets are pairwise disjoint (exact cover of a 15*n_rows subset).

    def _build_candidates(m):
        """Enumerate delta tuples with sum<=m; return list of (diffset, row, maxval)."""
        cands = []
        seen = set()
        def rec(idx, rem, cur):
            if idx == k:
                if rem >= 0:
                    row = _row_from_deltas(cur)
                    dset = frozenset(_row_diffs(row))
                    if dset not in seen:
                        seen.add(dset)
                        cands.append((dset, row, row[-1]))
                return
            for d in range(1, rem + 1):
                cur.append(d)
                rec(idx + 1, rem - d, cur)
                cur.pop()
        rec(0, m, [])
        return cands

    def _solve_exact_cover(cands, m, dl, seed):
        rng = random.Random(seed)
        nsets = len(cands)
        if nsets < n_rows:
            return None

        # cell -> list of candidate indices
        cell_to_cands = defaultdict(list)
        for si, (dset, row, mx) in enumerate(cands):
            for c in dset:
                cell_to_cands[c].append(si)

        used = set()
        chosen = []
        best_found = None

        def search(depth):
            nonlocal best_found
            if _elapsed() > dl:
                return False
            if depth == n_rows:
                best_found = chosen[:]
                return True
            # Most-constrained column: free cell with fewest disjoint candidates.
            best_cell = None
            best_count = None
            # candidate cells: min cell of each disjoint candidate
            for si in range(nsets):
                dset, row, mx = cands[si]
                if dset & used:
                    continue
                mc = min(dset)
                if mc in used:
                    continue
                cnt = 0
                for sj in cell_to_cands[mc]:
                    if not (cands[sj][0] & used):
                        cnt += 1
                if best_count is None or cnt < best_count:
                    best_count = cnt
                    best_cell = mc
                    if cnt <= 1:
                        break
                if _elapsed() > dl:
                    return False
            if best_cell is None:
                return False
            opts = []
            for sj in cell_to_cands[best_cell]:
                dset, row, mx = cands[sj]
                if dset & used:
                    continue
                opts.append((mx, sj))
            opts.sort()
            for _, sj in opts:
                dset, row, mx = cands[sj]
                used |= dset
                chosen.append(sj)
                if search(depth + 1):
                    return True
                chosen.pop()
                used -= dset
                if _elapsed() > dl:
                    return False
            return False

        if search(0):
            return [cands[i][1] for i in best_found]
        return None

    # ---------- Main ----------
    lb = _counting_lower_bound()

    best_rows = greedy_construct(0)
    best_scope = _scope(best_rows)
    best_rows = local_search_shrink(best_rows, deadline - 20, random.Random(1))
    best_scope = _scope(best_rows)

    min_m = max(lb, 50)
    attempt_m_list = []
    m = best_scope - 1
    while m >= min_m and len(attempt_m_list) < 30:
        attempt_m_list.append(m)
        m -= 1

    for m in attempt_m_list:
        if _elapsed() > deadline - 12:
            break
        t0 = time.time()
        try:
            cands = _build_candidates(m)
        except Exception:
            break
        if time.time() - t0 > deadline - 15:
            break
        if len(cands) < n_rows:
            continue
        found = None
        for seed in range(3):
            if _elapsed() > deadline - 10:
                break
            found = _solve_exact_cover(cands, m, deadline - 10, seed)
            if found is not None:
                break
        if found is not None:
            rows = [list(r) for r in found]
            if _valid(rows):
                scope = _scope(rows)
                if scope < best_scope:
                    best_rows = rows
                    best_scope = scope

    # Final polish
    if _elapsed() < deadline - 3:
        best_rows = local_search_shrink(best_rows, deadline - 2, random.Random(9))
        best_scope = _scope(best_rows)

    return best_rows
