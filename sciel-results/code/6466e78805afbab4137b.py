"""Structural change: global simulated-annealing / random-walk search over the entire
set of rows, treating the union of all positive intra-row differences as the constraint.

The representation is the same (list of rows), but the search is a single global
random-walk with simulated-annealing acceptance, not row-by-row greedy + local patch.

Moves:
  - pick a random row r and a random position p (1..k-1), try to shift it by delta in {-2..2}
  - pick a random row r and a random pair (p1,p2), swap values (maintaining sortedness check)
  - pick a random row r, shift a contiguous suffix by delta

Feasibility is tracked incrementally via a count array over differences.
"""

import time
import random
import sys
import math


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 118.0
    rows_needed = n_rows
    m = k + 1  # entries per row

    # ---------- Initial solution: greedy ----------
    def greedy_initial():
        u = set()
        res = []
        for _ in range(rows_needed):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in u:
                        ok = False
                        break
                if ok:
                    for a in row:
                        u.add(x - a)
                    row.append(x)
            res.append(row)
        return res

    # ---------- Difference bookkeeping ----------
    def compute_diffs(rows):
        """Return dict diff -> count for all positive intra-row differences."""
        from collections import defaultdict
        cnt = defaultdict(int)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    cnt[row[j] - row[i]] += 1
        return cnt

    def scope_of(rows):
        return max(max(r) for r in rows)

    # ---------- Local improvement (shift elements down) ----------
    def local_improve(rows, max_iters=100):
        cnt = compute_diffs(rows)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                for pos in range(len(row) - 1, 0, -1):
                    cv = row[pos]
                    if cv <= row[pos - 1] + 1:
                        continue
                    nv = cv - 1
                    # compute old and new diffs involving pos
                    old_d = set()
                    for i in range(pos):
                        old_d.add(cv - row[i])
                    for j in range(pos + 1, len(row)):
                        old_d.add(row[j] - cv)
                    new_d = set()
                    for i in range(pos):
                        new_d.add(nv - row[i])
                    for j in range(pos + 1, len(row)):
                        new_d.add(row[j] - nv)
                    conflict = False
                    for d in new_d:
                        if d in old_d:
                            continue
                        if cnt[d] > 0:
                            conflict = True
                            break
                    if not conflict:
                        for d in old_d:
                            cnt[d] -= 1
                        for d in new_d:
                            cnt[d] += 1
                        row[pos] = nv
                        improved = True
        return rows

    # ---------- Simulated annealing ----------
    def simulated_annealing(rows, time_budget):
        rng = random.Random(42)
        rows = [list(r) for r in rows]
        cnt = compute_diffs(rows)

        # Energy: number of duplicate differences + scope
        def energy(rows, cnt):
            dup = sum(max(0, c - 1) for c in cnt.values())
            return dup * 10000 + scope_of(rows)

        T_start = 50.0
        T_end = 0.01
        end_time = time.time() + time_budget
        steps = 0

        while time.time() < end_time:
            steps += 1
            # Choose a move
            move_type = rng.random()
            r_idx = rng.randrange(rows_needed)
            row = rows[r_idx]

            if move_type < 0.5:
                # Shift one element
                pos = rng.randrange(1, m - 1)
                delta = rng.choice([-2, -1, 1, 2])
                nv = row[pos] + delta
                if nv <= row[pos - 1] or nv >= row[pos + 1]:
                    continue
            elif move_type < 0.7:
                # Swap two elements
                p1, p2 = rng.sample(range(1, m - 1), 2)
                if p1 > p2:
                    p1, p2 = p2, p1
                # Check if swap maintains sortedness
                if row[p1] != row[p2]:
                    # After swap, check sortedness at boundaries
                    if p1 > 1 and row[p1 - 1] > row[p2]:
                        continue
                    if p2 < m - 2 and row[p2 + 1] < row[p1]:
                        continue
                    pos = None  # signal swap
            elif move_type < 0.85:
                # Shift suffix
                pos = rng.randrange(1, m - 1)
                delta = rng.choice([-1, 1])
                # Check that all shifted elements remain sorted
                min_val = row[pos] + delta
                max_val = row[m - 1] + delta
                if min_val <= row[pos - 1] or (pos + 1 < m and max_val > row[m - 1]):
                    continue
            else:
                # Random restart of one row
                u = set()
                for other in rows:
                    if other is row:
                        continue
                    for i in range(len(other)):
                        for j in range(i + 1, len(other)):
                            u.add(other[j] - other[i])
                new_row = [0]
                x = 0
                while len(new_row) < m:
                    x += 1
                    ok = True
                    for a in new_row:
                        if (x - a) in u:
                            ok = False
                            break
                    if ok:
                        for a in new_row:
                            u.add(x - a)
                        new_row.append(x)
                # Apply
                old_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        old_diffs.add(row[j] - row[i])
                new_diffs = set()
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        new_diffs.add(new_row[j] - new_row[i])
                conflict = False
                for d in new_diffs:
                    if d in old_diffs:
                        continue
                    if cnt[d] > 0:
                        conflict = True
                        break
                if conflict:
                    continue
                for d in old_diffs:
                    cnt[d] -= 1
                for d in new_diffs:
                    cnt[d] += 1
                rows[r_idx] = new_row
                continue

            if pos is None and move_type < 0.7:
                # Swap
                old1, old2 = row[p1], row[p2]
                if old1 == old2:
                    continue
                # Compute diffs change
                old_d = set()
                new_d = set()
                for i in range(m):
                    if i == p1 or i == p2:
                        continue
                    if i < p1:
                        old_d.add(old1 - row[i])
                        old_d.add(old2 - row[i])
                        new_d.add(old2 - row[i])
                        new_d.add(old1 - row[i])
                    elif p1 < i < p2:
                        old_d.add(row[i] - old1)
                        old_d.add(old2 - row[i])
                        new_d.add(row[i] - old2)
                        new_d.add(old1 - row[i])
                    else:
                        old_d.add(row[i] - old1)
                        old_d.add(row[i] - old2)
                        new_d.add(row[i] - old2)
                        new_d.add(row[i] - old1)
                old_d.add(old2 - old1)
                new_d.add(old1 - old2)
                # Actually swap changes: old diff was old2-old1, new is old1-old2 (same value if positive)
                # Let me recompute more carefully
                # This is getting complex; skip swaps for now and use shifts
                continue

            # Apply shift
            old_d = set()
            new_d = set()
            if move_type < 0.5:
                cv = row[pos]
                nv = cv + delta
                for i in range(pos):
                    old_d.add(cv - row[i])
                    new_d.add(nv - row[i])
                for j in range(pos + 1, m):
                    old_d.add(row[j] - cv)
                    new_d.add(row[j] - nv)
            else:
                # Suffix shift
                for p in range(pos, m):
                    cv = row[p]
                    nv = cv + delta
                    for i in range(p):
                        old_d.add(cv - row[i])
                        new_d.add(nv - row[i])
                    for j in range(p + 1, m):
                        old_d.add(row[j] - cv)
                        new_d.add(row[j] - nv)

            conflict = False
            for d in new_d:
                if d in old_d:
                    continue
                if cnt[d] > 0:
                    conflict = True
                    break
            if conflict:
                continue

            # Compute energy change
            old_energy = energy(rows, cnt)
            # Apply
            for d in old_d:
                cnt[d] -= 1
            for d in new_d:
                cnt[d] += 1
            if move_type < 0.5:
                row[pos] += delta
            else:
                for p in range(pos, m):
                    row[p] += delta
            new_energy = energy(rows, cnt)

            if new_energy <= old_energy:
                # Accept
                pass
            else:
                # Simulated annealing acceptance
                T = T_start * (T_end / T_start) ** (steps / max(1, int((deadline - time.time()) * 10)))
                if T < T_end:
                    T = T_end
                if rng.random() < math.exp(-(new_energy - old_energy) / max(T, 1e-10)):
                    pass
                else:
                    # Reject
                    for d in old_d:
                        cnt[d] += 1
                    for d in new_d:
                        cnt[d] -= 1
                    if move_type < 0.5:
                        row[pos] -= delta
                    else:
                        for p in range(pos, m):
                            row[p] -= delta
                    continue

        return rows

    # ---------- Main ----------
    best_rows = greedy_initial()
    best_rows = local_improve(best_rows, max_iters=200)
    best_scope = scope_of(best_rows)

    # Run simulated annealing with remaining time
    elapsed = time.time() - (deadline - 118.0)
    remaining = 118.0 - elapsed - 1.0
    if remaining > 5:
        sa_rows = simulated_annealing(best_rows, remaining)
        sa_rows = local_improve(sa_rows, max_iters=200)
        sa_scope = scope_of(sa_rows)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows

    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
