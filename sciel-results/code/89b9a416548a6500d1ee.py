"""Difference triangle set (4,7): minimize the largest entry across rows."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 108.0  # safety margin
    rng = random.Random(7)

    def diffs_of(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def valid(rows):
        used = set()
        for row in rows:
            for d in diffs_of(row):
                if d in used:
                    return False
                used.add(d)
        return True

    def max_entry(rows):
        return max(r[-1] for r in rows)

    def greedy_row(used, k, cap, horizon=6):
        """Build a row starting at 0 with k+1 increasing ints, all <= cap.
        At each step consider candidate next-marks in [x+1, x+horizon] that are
        valid; pick the one that minimizes a look-ahead estimate of final endpoint.
        """
        row = [0]
        x = 0
        while len(row) < k + 1:
            best_cand = None
            best_key = None
            for cand in range(x + 1, min(x + 1 + horizon, cap + 1)):
                new = {cand - a for a in row}
                if new & used:
                    continue
                # look-ahead: tentatively extend and see how far we can go
                trow = row + [cand]
                tused = used | new
                tx = cand
                ok = True
                while len(trow) < k + 1:
                    found = None
                    for c2 in range(tx + 1, cap + 1):
                        n2 = {c2 - a for a in trow}
                        if not (n2 & tused):
                            found = c2
                            break
                    if found is None:
                        ok = False
                        break
                    trow.append(found)
                    tused |= {found - a for a in trow[:-1]}
                    tx = found
                if ok:
                    # key: final endpoint (smaller better); tie-break by cand
                    key = (trow[-1], cand)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_cand = cand
            if best_cand is None:
                # fall back to smallest valid within cap
                for cand in range(x + 1, cap + 1):
                    new = {cand - a for a in row}
                    if not (new & used):
                        best_cand = cand
                        break
                if best_cand is None:
                    return None
            row.append(best_cand)
            used |= {best_cand - a for a in row[:-1]}
            x = best_cand
        return row

    def run_greedy(caps, horizon=6):
        used = set()
        rows = []
        for c in caps:
            row = greedy_row(used, k, c, horizon=horizon)
            if row is None:
                return None
            used |= diffs_of(row)
            rows.append(row)
        return rows

    # Phase 1: baseline with a generous cap
    base_cap = 200
    best = run_greedy([base_cap] * n_rows, horizon=4)
    if best is None:
        best = [[i * (k + 1) for i in range(k + 1)] for _ in range(n_rows)]
        return best
    best_score = max_entry(best)

    # Phase 2: descend the global cap M. Try to build all rows with cap M.
    # Use a few horizons and a few random row orders to escape local traps.
    M = best_score - 1
    while M > k and time.time() < deadline:
        success = False
        for horizon in (2, 4, 6):
            if time.time() > deadline:
                break
            for _ in range(25):
                if time.time() > deadline:
                    break
                caps = [M] * n_rows
                res = run_greedy(caps, horizon=horizon)
                if res is not None and valid(res) and max_entry(res) <= M:
                    best = res
                    best_score = M
                    success = True
                    break
            if success:
                break
        if success:
            M = best_score - 1
        else:
            break
    return best
# EVOLVE-BLOCK-END
