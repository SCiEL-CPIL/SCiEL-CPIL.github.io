import time
import random

def construct(n_rows, k):
    deadline = time.time() + 115.0
    floor = n_rows * (k * (k + 1) // 2)
    INIT_CAP = max(600, floor * 4)
    NODE_CAP = 20000
    rng = random.Random(20250101)

    best_rows = None
    best_scope = 10**18

    def make_greedy(used, max_last, randomize):
        row = [0]
        diffs = set()
        nodes = 0
        cap = max_last if max_last is not None else INIT_CAP
        def bt(pos):
            nonlocal nodes
            nodes += 1
            if nodes > NODE_CAP:
                return False
            if pos == k + 1:
                return True
            start = row[-1] + 1
            end = cap
            cands = []
            for x in range(start, end + 1):
                new = {x - a for a in row}
                if (new & used) or (new & diffs):
                    continue
                cands.append((x, new))
                if randomize and len(cands) >= 8:
                    break
            if not cands:
                return False
            if randomize:
                rng.shuffle(cands)
            for x, new in cands:
                row.append(x)
                diffs |= new
                if bt(pos + 1):
                    return True
                row.pop()
                diffs -= new
            return False
        if bt(1):
            return row, diffs
        return None, None

    def row_diff_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def run_local_search(rows, used, row_diffs, budget):
        nonlocal best_scope, best_rows
        t_end = time.time() + budget
        scope = max(r[-1] for r in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
        stagnant = 0
        MAX_STAG = 50
        while time.time() < t_end and scope > floor and stagnant < MAX_STAG:
            M = scope
            improved = False
            max_rows = [i for i, r in enumerate(rows) if r[-1] == M]
            rng.shuffle(max_rows)
            for i in max_rows:
                if time.time() > t_end:
                    break
                if rows[i][-1] != M:
                    continue
                old_diffs = row_diffs[i]
                used -= old_diffs
                target = max(M - 1, floor)
                new_row, new_diffs = make_greedy(used, target, True)
                if new_row is not None:
                    rows[i] = new_row
                    row_diffs[i] = new_diffs
                    used |= new_diffs
                    ns = max(r[-1] for r in rows)
                    if ns < scope:
                        scope = ns
                        improved = True
                        if scope < best_scope:
                            best_scope = scope
                            best_rows = [r[:] for r in rows]
                else:
                    used |= old_diffs
                    row_diffs[i] = old_diffs
            if improved:
                stagnant = 0
            else:
                stagnant += 1
                # packing / shuffle moves on random rows to escape local minima
                for _ in range(5):
                    if time.time() > t_end:
                        break
                    i = rng.randrange(n_rows)
                    old_diffs = row_diffs[i]
                    old_last = rows[i][-1]
                    used -= old_diffs
                    target = max(old_last - 1, 17)
                    new_row, new_diffs = make_greedy(used, target, True)
                    if new_row is not None:
                        rows[i] = new_row
                        row_diffs[i] = new_diffs
                        used |= new_diffs
                        ns = max(r[-1] for r in rows)
                        if ns < scope:
                            scope = ns
                            improved = True
                            stagnant = 0
                            if scope < best_scope:
                                best_scope = scope
                                best_rows = [r[:] for r in rows]
                    else:
                        used |= old_diffs
                        row_diffs[i] = old_diffs
        return scope

    # Phase 1: deterministic greedy init (matches seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row, diffs = make_greedy(used, None, False)
        if row is None:
            row, diffs = make_greedy(used, None, True)
        if row is None:
            # extreme fallback
            row = list(range(k + 1))
            diffs = row_diff_set(row)
        rows.append(row)
        used |= diffs
    row_diffs = [row_diff_set(r) for r in rows]
    best_scope = max(r[-1] for r in rows)
    best_rows = [r[:] for r in rows]

    # Phase 2a: deep local search on the strong init
    remaining = deadline - time.time()
    run_local_search(rows, used, row_diffs, min(45.0, max(5.0, remaining - 5.0)))

    # Phase 2b: randomized restarts
    while time.time() < deadline - 3.0 and best_scope > floor:
        used = set()
        rows = []
        ok = True
        for _ in range(n_rows):
            row, diffs = make_greedy(used, None, True)
            if row is None:
                ok = False
                break
            rows.append(row)
            used |= diffs
        if not ok:
            continue
        row_diffs = [row_diff_set(r) for r in rows]
        run_local_search(rows, used, row_diffs, min(6.0, max(1.5, deadline - time.time() - 3.0)))

    return best_rows
