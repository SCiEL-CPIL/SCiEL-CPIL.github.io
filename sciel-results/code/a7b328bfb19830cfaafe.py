import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s

    # Each row has k+1 marks starting at 0, strictly increasing.
    # Valid iff all positive intra-row differences are distinct across all rows.
    # Total differences per row = k*(k+1)//2 ; across n_rows rows.

    def row_diffs(row):
        # returns set of positive differences for a row (list of k+1 ints, row[0]==0)
        d = set()
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                d.add(row[j] - ai)
        return d

    def is_valid(rows):
        seen = set()
        for r in rows:
            for i in range(len(r)):
                ai = r[i]
                for j in range(i + 1, len(r)):
                    v = r[j] - ai
                    if v in seen:
                        return False
                    seen.add(v)
        return True

    # ---- deterministic greedy baseline (order-dependent) ----
    def greedy(rows_order=None, rng=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            rows.append(row)
        return rows

    best = greedy()
    best_scope = max(max(r) for r in best)

    # ---- randomized local search ----
    # Moves:
    #  1) pick a row and a position, try to replace mark with a smaller/larger value
    #     keeping row strictly increasing and all diffs globally unique.
    #  2) pick a row and shift all marks after index i by +/-1 if valid.
    #  3) rebuild one row greedily given current used set (reinsert).
    # Objective: minimize max mark (scope).

    rng = random.Random(12345)

    def scope_of(rows):
        return max(max(r) for r in rows)

    def try_improve(rows, rng, deadline):
        # returns improved rows or None
        n = len(rows)
        m = len(rows[0])  # k+1
        used = set()
        for r in rows:
            for i in range(m):
                ai = r[i]
                for j in range(i + 1, m):
                    used.add(r[j] - ai)

        def remove_row_diffs(r):
            for i in range(m):
                ai = r[i]
                for j in range(i + 1, m):
                    used.discard(r[j] - ai)

        def add_row_diffs(r):
            for i in range(m):
                ai = r[i]
                for j in range(i + 1, m):
                    used.add(r[j] - ai)

        improved = False
        # hill-climb: repeatedly try small edits that reduce scope or are neutral
        for _ in range(2000):
            if time.time() > deadline - 5.0:
                break
            ri = rng.randrange(n)
            r = rows[ri]
            # pick a position to modify (not 0)
            pi = rng.randrange(1, m)
            old = r[pi]

            # candidate new values: try smaller first (to reduce scope), then larger
            lo = r[pi - 1] + 1
            hi = r[pi + 1] - 1 if pi + 1 < m else old + 50

            # remove this row's diffs from used, then test candidates
            remove_row_diffs(r)
            found = False
            # try values in increasing distance from old, preferring lower
            candidates = []
            # smaller side
            for v in range(lo, old):
                candidates.append(v)
            # larger side (limited)
            for v in range(old + 1, hi + 1):
                candidates.append(v)

            rng.shuffle(candidates)
            for v in candidates:
                if v <= lo or (pi + 1 < m and v >= hi):
                    continue
                # tentatively set
                r[pi] = v
                ok = True
                # check intra-row monotonic (already by range) and global diff uniqueness
                for i in range(m):
                    ai = r[i]
                    for j in range(i + 1, m):
                        d = r[j] - ai
                        if d in used:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    # accept if it reduces scope or is a random walk
                    new_scope = max(max(x) for x in rows)
                    if new_scope < best_scope or rng.random() < 0.15:
                        add_row_diffs(r)
                        found = True
                        improved = True
                        break
                # revert
                r[pi] = old
            if not found:
                # restore diffs for original row
                add_row_diffs(r)
                r[pi] = old
            else:
                r[pi] = old  # already set to v? we set r[pi]=v in loop; keep v
                # Actually in loop we set r[pi]=v and broke; do not revert.
                # Re-set properly:
                # (no-op; r[pi] currently v)
        # recompute best scope
        s = scope_of(rows)
        if s < best_scope:
            best_scope = s
        return rows, improved

    # Multi-start: different random orders / random initial builds
    for start in range(200):
        if time.time() > deadline - 10.0:
            break
        # random greedy: shuffle which "slot" we fill? Instead randomize by
        # building rows in a random permutation of row indices doesn't change
        # greedy result. So randomize by adding small random offsets during build.
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    # random skip to diversify
                    if rng.random() < 0.2:
                        # try a few extra x before accepting
                        for _extra in range(rng.randrange(1, 4)):
                            x += 1
                            ok2 = True
                            for a in row:
                                if (x - a) in used:
                                    ok2 = False
                                    break
                            if not ok2:
                                x -= 1
                                break
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            rows.append(row)
        s = scope_of(rows)
        if s < best_scope:
            best = rows
            best_scope = s
        # local search on this candidate
        if time.time() > deadline - 8.0:
            break
        rows2, _ = try_improve(rows, rng, deadline)
        s2 = scope_of(rows2)
        if s2 < best_scope:
            best = rows2
            best_scope = s2

    # final validation
    if not is_valid(best):
        # fallback to deterministic greedy
        best = greedy()

    return best
# EVOLVE-BLOCK-END
