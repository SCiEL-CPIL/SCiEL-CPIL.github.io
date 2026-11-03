import time
import random
import math
import sys

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 145.0
    row_len = k + 1
    n_diffs_per_row = k * (k + 1) // 2
    sys.setrecursionlimit(100000)

    def row_mask(row):
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= (1 << (row[j] - ri))
        return m

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != row_len:
                return False
            if row[0] != 0:
                return False
            prev = -1
            for v in row:
                if v <= prev:
                    return False
                prev = v
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # ---------- Fast row construction (greedy) ----------
    def build_row_greedy(used_mask, max_scope, rng, randomize=False, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        row = [0]
        local_mask = 0
        for _ in range(1, row_len):
            last = row[-1]
            found = False
            start = last + 1
            if randomize and rng.random() < 0.25:
                start = last + rng.randint(1, 2)
            # cap: we must still be able to fit remaining entries
            remaining = row_len - len(row)
            upper = max_scope - (remaining - 1)
            if upper < start:
                return None
            for t in range(start, upper + 1):
                nm = 0
                for a in row:
                    nm |= (1 << (t - a))
                if (nm & used_mask) or (nm & local_mask):
                    continue
                row.append(t)
                local_mask |= nm
                found = True
                break
            if not found:
                return None
        return row

    # ---------- Phase 1: Greedy construction with many seeds ----------
    best_rows = None
    best_scope = float('inf')
    rng = random.Random(12345)

    # Try many random seeds; deterministic first few, then random
    seeds_to_try = list(range(40)) + [rng.randint(0, 10**9) for _ in range(60)]
    for seed in seeds_to_try:
        if time.time() >= DEADLINE - 40:
            break
        r = random.Random(seed)
        used_mask = 0
        rows = []
        ok = True
        for _ in range(n_rows):
            res = build_row_greedy(used_mask, 200, r, randomize=(seed >= 40))
            if res is None:
                ok = False
                break
            rows.append(res)
            used_mask |= row_mask(res)
        if ok:
            sc = scope_of(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [list(x) for x in rows]

    if best_rows is None:
        # Fallback: pure greedy, no randomization
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            res = build_row_greedy(used_mask, 500, random.Random(0))
            if res is None:
                # absolute fallback: linear
                row = [0]
                for pos in range(1, k + 1):
                    v = row[-1] + 1
                    while True:
                        ok2 = True
                        for prev in row:
                            if (v - prev) in used_mask:
                                ok2 = False
                                break
                        if ok2:
                            row.append(v)
                            for prev in row[:-1]:
                                used_mask |= (1 << (v - prev))
                            break
                        v += 1
                rows.append(row)
            else:
                rows.append(res)
                used_mask |= row_mask(res)
        return rows

    # ---------- Phase 2: Local search (row-rebuild + entry-shift) ----------
    cur_rows = [list(x) for x in best_rows]
    cur_scope = best_scope
    best_rows = [list(x) for x in cur_rows]
    best_scope = cur_scope

    # Precompute per-row masks for fast used_mask computation
    cur_masks = [row_mask(r) for r in cur_rows]

    def used_mask_except(idx):
        um = 0
        for i in range(n_rows):
            if i != idx:
                um |= cur_masks[i]
        return um

    T = 3.0
    cooling = 0.9997
    min_T = 0.05
    iters = 0
    max_iters = 200000

    while time.time() < DEADLINE - 15 and T > min_T and iters < max_iters:
        iters += 1
        r = rng.random()

        if r < 0.6:
            # Row rebuild: pick a row, rebuild it greedily with randomization
            i = rng.randint(0, n_rows - 1)
            old_row = list(cur_rows[i])
            old_mask = cur_masks[i]
            um = used_mask_except(i)
            cap = cur_scope + 2
            new_row = build_row_greedy(um, cap, rng, randomize=True)
            if new_row is None or new_row[-1] >= old_row[-1]:
                continue
            new_mask = row_mask(new_row)
            if new_mask & um:
                continue
            # Accept (improvement)
            cur_rows[i] = new_row
            cur_masks[i] = new_mask
            cur_scope = scope_of(cur_rows)
            if cur_scope < best_scope:
                best_scope = cur_scope
                best_rows = [list(x) for x in cur_rows]
        else:
            # Entry shift: pick a row and an entry, try small shifts
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, row_len - 1)
            old_row = list(cur_rows[i])
            old_val = old_row[j]
            um = used_mask_except(i)
            improved = False
            # Try to reduce the value
            for delta in range(-1, -6, -1):
                nv = old_val + delta
                if nv <= old_row[j - 1]:
                    continue
                if j < row_len - 1 and nv >= old_row[j + 1]:
                    continue
                new_row = list(old_row)
                new_row[j] = nv
                nm = row_mask(new_row)
                if nm & um:
                    continue
                cur_rows[i] = new_row
                cur_masks[i] = nm
                cur_scope = scope_of(cur_rows)
                if cur_scope < best_scope:
                    best_scope = cur_scope
                    best_rows = [list(x) for x in cur_rows]
                improved = True
                break
            if not improved:
                # Try small random shifts
                for _ in range(3):
                    delta = rng.randint(-4, 4)
                    if delta == 0:
                        continue
                    nv = old_val + delta
                    if nv <= old_row[j - 1]:
                        continue
                    if j < row_len - 1 and nv >= old_row[j + 1]:
                        continue
                    new_row = list(old_row)
                    new_row[j] = nv
                    nm = row_mask(new_row)
                    if nm & um:
                        continue
                    cur_rows[i] = new_row
                    cur_masks[i] = nm
                    cur_scope = scope_of(cur_rows)
                    if cur_scope < best_scope:
                        best_scope = cur_scope
                        best_rows = [list(x) for x in cur_rows]
                    break

        T *= cooling

    # ---------- Phase 3: Scope-reduction compression ----------
    # Try to reduce the global scope by finding rows with smaller max
    cur_rows = [list(x) for x in best_rows]
    cur_scope = scope_of(cur_rows)
    best_rows = [list(x) for x in cur_rows]
    best_scope = cur_scope
    cur_masks = [row_mask(r) for r in cur_rows]

    improved = True
    while improved and time.time() < DEADLINE - 2:
        improved = False
        for target_scope in range(cur_scope - 1, max(cur_scope - 20, 0), -1):
            if time.time() >= DEADLINE - 2:
                break
            for i in range(n_rows):
                if time.time() >= DEADLINE - 2:
                    break
                if cur_rows[i][-1] <= target_scope:
                    continue
                um = 0
                for ri in range(n_rows):
                    if ri != i:
                        um |= cur_masks[ri]
                # Try to build a row with max <= target_scope
                for attempt in range(8):
                    new_row = build_row_greedy(um, target_scope, rng, randomize=True,
                                               seed=attempt * 7919 + i)
                    if new_row is not None and new_row[-1] <= target_scope:
                        nm = row_mask(new_row)
                        if nm & um:
                            continue
                        cur_rows[i] = new_row
                        cur_masks[i] = nm
                        cur_scope = scope_of(cur_rows)
                        if cur_scope < best_scope:
                            best_scope = cur_scope
                            best_rows = [list(x) for x in cur_rows]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # ---------- Final validation ----------
    if not is_valid(best_rows):
        # Try to fix by rebuilding any problematic row
        for i in range(n_rows):
            um = 0
            for ri in range(n_rows):
                if ri != i:
                    um |= row_mask(best_rows[ri])
            new_row = build_row_greedy(um, best_scope + 10, rng, randomize=True)
            if new_row is not None:
                best_rows[i] = new_row
        if not is_valid(best_rows):
            # Last resort: full rebuild
            used_mask = 0
            rows = []
            for _ in range(n_rows):
                res = build_row_greedy(used_mask, 300, rng, randomize=True)
                if res is None:
                    res = build_row_greedy(used_mask, 500, random.Random(0))
                if res is not None:
                    rows.append(res)
                    used_mask |= row_mask(res)
                else:
                    row = [0]
                    for pos in range(1, k + 1):
                        v = row[-1] + 1
                        while True:
                            ok2 = True
                            for prev in row:
                                if (1 << (v - prev)) & used_mask:
                                    ok2 = False
                                    break
                            if ok2:
                                row.append(v)
                                for prev in row[:-1]:
                                    used_mask |= (1 << (v - prev))
                                break
                            v += 1
                    rows.append(row)
            best_rows = rows

    return best_rows
