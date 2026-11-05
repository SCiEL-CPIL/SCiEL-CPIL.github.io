"""Difference triangle set via local search on rows."""
import time
import random

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    deadline = time.time() + 140.0
    rng = random.Random(98765)

    def greedy(seed_rng=None):
        used, rows = set(), []
        r = seed_rng if seed_rng is not None else rng
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)

    def other_used(rows, exclude):
        used = set()
        for i, row in enumerate(rows):
            if i == exclude:
                continue
            for a in row:
                for b in row:
                    if a < b:
                        used.add(b - a)
        return used

    def rebuild_row(used_other, k, deadline, perturb=False, rng=rng):
        row, x = [0], 0
        while len(row) < k + 1:
            if time.time() > deadline:
                return None
            x += 1
            new = {x - a for a in row}
            if new & used_other:
                continue
            if perturb and rng.random() < 0.1:
                continue
            row.append(x)
        if len(row) == k + 1:
            return row
        return None

    def try_shrink_row(cur, idx, deadline):
        used_other = other_used(cur, idx)
        new_row = rebuild_row(used_other, k, deadline, perturb=False)
        if new_row is None:
            return False
        if max(new_row) < max(cur[idx]):
            cur[idx] = new_row
            return True
        return False

    cur = [r[:] for r in rows]
    cur_max = max(max(r) for r in cur)
    no_improve = 0

    while time.time() < deadline:
        improved = False
        order = list(range(n_rows))
        rng.shuffle(order)
        for idx in order:
            if time.time() > deadline:
                break
            if try_shrink_row(cur, idx, deadline):
                improved = True
                no_improve = 0
                cur_max = max(max(r) for r in cur)
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [r[:] for r in cur]
        if not improved:
            no_improve += 1
            idx = rng.randrange(n_rows)
            used_other = other_used(cur, idx)
            for _ in range(5):
                new_row = rebuild_row(used_other, k, deadline, perturb=True)
                if new_row is not None:
                    rd = set()
                    for a in new_row:
                        for b in new_row:
                            if a < b:
                                rd.add(b - a)
                    if not (rd & used_other):
                        cur[idx] = new_row
                        cur_max = max(max(r) for r in cur)
                        if cur_max < best_max:
                            best_max = cur_max
                            best_rows = [r[:] for r in cur]
                        no_improve = 0
                        break
        if no_improve > 300:
            no_improve = 0
            cur = greedy()
            cur_max = max(max(r) for r in cur)

    return best_rows

# EVOLVE-BLOCK-END
