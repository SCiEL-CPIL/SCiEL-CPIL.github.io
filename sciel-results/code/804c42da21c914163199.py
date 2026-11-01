"""Difference triangle set: minimize largest entry via greedy init + SA."""
import time

def construct(n_rows, k):
    deadline = time.time() + 140.0
    n_diffs = k * (k + 1) // 2
    # Precompute the difference offsets for each position in a row:
    # diff_offsets[j] = list of (i, j) pairs? We need, for a row, the set of
    # differences a_j - a_i for all i<j. Represent a row by its values; the
    # multiset of differences is fixed by positions.
    # Precompute, for each index j (1..k), the list of earlier indices i<j.
    pos_pairs = []
    for j in range(1, k + 1):
        pos_pairs.append(list(range(j)))

    def row_diffs(row):
        # return list of all positive differences (a_j - a_i, i<j)
        out = []
        for j in range(1, k + 1):
            aj = row[j]
            for i in range(j):
                out.append(aj - row[i])
        return out

    def build_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = [x - a for a in row]
                # check none already used
                ok = True
                for d in new:
                    if d in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for d in new:
                        used.add(d)
            rows.append(row)
        return rows

    best_rows = build_greedy()
    best_max = max(max(r) for r in best_rows)

    # Simulated annealing on the whole set of rows.
    # State: list of rows (each list of k+1 ints starting at 0).
    # Move: pick a random row, pick a random position (1..k), perturb its value
    #   by a small delta, ensuring strictly increasing and 0 start.
    # Cost: max entry (we want to minimize). Also must keep all differences distinct.
    # We track the global used-difference set; a move is valid only if the
    #   differences of the changed row don't collide with others.

    import random
    rng = random.Random(12345)

    def compute_used(rows):
        used = set()
        for r in rows:
            for j in range(1, k + 1):
                aj = r[j]
                for i in range(j):
                    used.add(aj - r[i])
        return used

    def row_diff_set(r):
        s = set()
        for j in range(1, k + 1):
            aj = r[j]
            for i in range(j):
                s.add(aj - r[i])
        return s

    rows = [list(r) for r in best_rows]
    used = compute_used(rows)
    cur_max = best_max

    T = max(1.0, cur_max * 0.05)
    T_min = 0.05
    cool = 0.9999

    steps = 0
    while time.time() < deadline:
        # batch of steps per time check
        for _ in range(200):
            if time.time() >= deadline:
                break
            steps += 1
            ri = rng.randrange(n_rows)
            row = rows[ri]
            # choose a position to move
            p = rng.randrange(1, k + 1)
            # choose delta
            delta = rng.randint(-3, 3)
            if delta == 0:
                continue
            new_val = row[p] + delta
            # check increasing: row[p-1] < new_val < row[p+1] (if exists)
            if new_val <= row[p - 1]:
                continue
            if p < k and new_val >= row[p + 1]:
                continue
            # old diffs of this row
            old_d = row_diff_set(row)
            # candidate row
            row[p] = new_val
            new_d = row_diff_set(row)
            row[p] = row[p] - delta  # undo
            # valid if new_d doesn't collide with (used - old_d)
            others = used - old_d
            if new_d & others:
                continue
            # accept?
            new_max = max(row)
            cost = new_max
            cur_cost = cur_max
            if cost < cur_cost:
                accept = True
            else:
                import math
                if T > 0:
                    accept = rng.random() < math.exp(-(cost - cur_cost) / T)
                else:
                    accept = False
            if accept:
                # commit
                row[p] = new_val
                used = (used - old_d) | new_d
                if new_max > cur_max:
                    cur_max = new_max
                elif new_max == cur_max:
                    # recompute exact max to be safe
                    cur_max = max(max(r) for r in rows)
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(r) for r in rows]
            T *= cool
            if T < T_min:
                T = T_min
        # recompute cur_max periodically to avoid drift
        if steps % 5000 < 200:
            cur_max = max(max(r) for r in rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(r) for r in rows]

    # Final validation
    used_final = set()
    for r in best_rows:
        for j in range(1, k + 1):
            aj = r[j]
            for i in range(j):
                d = aj - r[i]
                if d in used_final:
                    # invalid; fall back to greedy
                    return build_greedy()
                used_final.add(d)
    return best_rows
