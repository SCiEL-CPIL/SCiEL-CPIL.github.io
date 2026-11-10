import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 148  # safety margin before 150s kill

    total_diffs = k * (k + 1) // 2  # per row
    counting_lb = n_rows * total_diffs

    def row_diffs(row):
        s = set()
        L = len(row)
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                s.add(row[j] - ri)
        return s

    def is_valid(rows):
        all_d = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            d = row_diffs(row)
            if d & all_d:
                return False
            all_d |= d
        return True

    def greedy_row(forbidden, max_val_hint):
        row = [0]
        local = set()
        upper = max_val_hint if max_val_hint else 2000
        for _ in range(k):
            base = row[-1]
            found = False
            for v in range(base + 1, upper + 1):
                ok = True
                for a in row:
                    d = v - a
                    if d in forbidden or d in local:
                        ok = False
                        break
                if ok:
                    row.append(v)
                    for a in row[:-1]:
                        local.add(row[-1] - a)
                    found = True
                    break
            if not found:
                return None
        return row

    def initial_construct(max_hint):
        rows = []
        used = set()
        for i in range(n_rows):
            r = greedy_row(used, max_hint)
            if r is None:
                return None
            rows.append(r)
            used |= row_diffs(r)
        return rows

    # Try to get an initial solution
    hint = max(counting_lb + 10, 130)
    init_rows = initial_construct(hint)
    if init_rows is None:
        hint = counting_lb + 60
        init_rows = initial_construct(hint)
    if init_rows is None:
        hint = counting_lb + 200
        init_rows = initial_construct(hint)
    if init_rows is None:
        rows = []
        used = set()
        for i in range(n_rows):
            row = [0]
            local = set()
            for _ in range(k):
                base = row[-1]
                v = base + 1
                while True:
                    ok = True
                    for a in row:
                        d = v - a
                        if d in used or d in local:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                row.append(v)
                for a in row[:-1]:
                    local.add(v - a)
            rows.append(row)
            used |= row_diffs(row)
        init_rows = rows

    def score_of(rows):
        return max(max(r) for r in rows)

    best_rows = [r[:] for r in init_rows]
    best_score = score_of(best_rows)

    # --- Optimized Backtracking with explicit state passing ---
    def backtrack_solve(max_val):
        """Try to construct n_rows valid rows with all entries <= max_val.
        Uses iterative deepening with explicit state to minimize overhead."""
        if time.time() > deadline:
            return None

        # Precompute: for each position in a row, what's the max possible value
        # We build rows one at a time, trying to keep entries small

        rows = []
        used_diffs = set()
        n = n_rows
        kk = k

        def try_build_row(row, pos, local_diffs):
            """Try to complete row from position pos. Returns True on success."""
            if time.time() > deadline:
                return False
            if pos == kk:
                return True

            base = row[-1]
            # Try candidate values, starting from smallest
            for v in range(base + 1, max_val + 1):
                # Check all differences
                ok = True
                for a in row:
                    d = v - a
                    if d in used_diffs or d in local_diffs:
                        ok = False
                        break
                if not ok:
                    continue

                # Add v to row
                row.append(v)
                new_diffs = []
                for a in row[:-1]:
                    d = v - a
                    new_diffs.append(d)
                local_diffs.update(new_diffs)

                if try_build_row(row, pos + 1, local_diffs):
                    return True

                # Backtrack
                row.pop()
                for d in new_diffs:
                    local_diffs.discard(d)

            return False

        def solve(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True

            # Try to build row idx
            row = [0]
            local_diffs = set()

            if try_build_row(row, 0, local_diffs):
                rows.append(row)
                rd = row_diffs(row)
                used_diffs.update(rd)
                if solve(idx + 1):
                    return True
                else:
                    used_diffs.difference_update(rd)
                    rows.pop()

            return False

        if solve(0):
            return rows
        return None

    # --- Simulated Annealing with improved moves ---
    def greedy_candidates(forbidden, row, max_scan=12):
        cands = []
        base = row[-1]
        L = len(row)
        local = set()
        for i in range(L):
            for j in range(i + 1, L):
                local.add(row[j] - row[i])
        scan_limit = base + 600
        for v in range(base + 1, scan_limit + 1):
            ok = True
            for a in row:
                d = v - a
                if d in forbidden or d in local:
                    ok = False
                    break
            if ok:
                cands.append(v)
                if len(cands) >= max_scan:
                    break
        return cands

    def build_row_with_skips(forbidden, skips, max_hint):
        row = [0]
        local = set()
        for t in range(k):
            cands = greedy_candidates(forbidden, row, 12)
            if not cands:
                base = row[-1]
                L = len(row)
                local2 = set()
                for i in range(L):
                    for j in range(i + 1, L):
                        local2.add(row[j] - row[i])
                v = base + 1
                while True:
                    ok = True
                    for a in row:
                        d = v - a
                        if d in forbidden or d in local2:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                cands = [v]
            idx = skips[t]
            if idx < 0:
                idx = 0
            if idx >= len(cands):
                idx = 0
            v = cands[idx]
            row.append(v)
            for a in row[:-1]:
                local.add(v - a)
        if row[-1] > max_hint:
            return None
        return row

    cur_rows = [r[:] for r in init_rows]
    cur_row_diffs = [row_diffs(r) for r in cur_rows]
    cur_score = score_of(cur_rows)

    T = 20.0
    cooling = 0.9993
    min_T = 0.03
    # Diverse skip choices for wider exploration
    skip_choices = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    def sample_skips():
        return [random.choice(skip_choices) for _ in range(k)]

    # Try backtracking for target values below current best
    for target in range(best_score - 1, counting_lb - 1, -1):
        if time.time() > deadline - 40:
            break
        sol = backtrack_solve(target)
        if sol is not None:
            if is_valid(sol):
                best_rows = sol
                best_score = target
                # Update current state
                cur_rows = [r[:] for r in sol]
                cur_row_diffs = [row_diffs(r) for r in cur_rows]
                cur_score = best_score
                break

    # Continue with simulated annealing
    sa_steps = 0
    while time.time() < deadline - 5:
        if T < min_T:
            T = 15.0
            # Restart with fresh temperature if stuck
            if random.random() < 0.3:
                # Try a small perturbation
                idx = random.randrange(n_rows)
                forbidden = set()
                for i in range(n_rows):
                    if i != idx:
                        forbidden |= cur_row_diffs[i]
                skips = sample_skips()
                max_hint = best_score + 10
                new_row = build_row_with_skips(forbidden, skips, max_hint)
                if new_row is not None:
                    new_row_d = row_diffs(new_row)
                    if not (new_row_d & forbidden):
                        cur_rows[idx] = new_row
                        cur_row_diffs[idx] = new_row_d
                        cur_score = score_of(cur_rows)
                        if cur_score < best_score:
                            best_score = cur_score
                            best_rows = [r[:] for r in cur_rows]

        idx = random.randrange(n_rows)

        forbidden = set()
        for i in range(n_rows):
            if i != idx:
                forbidden |= cur_row_diffs[i]

        skips = sample_skips()
        max_hint = best_score + 8
        new_row = build_row_with_skips(forbidden, skips, max_hint)
        if new_row is None:
            T *= cooling
            sa_steps += 1
            continue

        new_row_d = row_diffs(new_row)
        if new_row_d & forbidden:
            T *= cooling
            sa_steps += 1
            continue

        new_rows = cur_rows[:]
        new_rows[idx] = new_row
        ns = score_of(new_rows)

        delta = ns - cur_score
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if random.random() < 0.08:
                accept = True
        else:
            prob = math.exp(-delta / T)
            if random.random() < prob:
                accept = True

        if accept:
            cur_rows[idx] = new_row
            cur_row_diffs[idx] = new_row_d
            cur_score = ns
            if cur_score < best_score:
                best_score = cur_score
                best_rows = [r[:] for r in cur_rows]

        T *= cooling
        sa_steps += 1

    if not is_valid(best_rows):
        rows = []
        used = set()
        for i in range(n_rows):
            row = [0]
            local = set()
            for _ in range(k):
                base = row[-1]
                v = base + 1
                while True:
                    ok = True
                    for a in row:
                        d = v - a
                        if d in used or d in local:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                row.append(v)
                for a in row[:-1]:
                    local.add(v - a)
            rows.append(row)
            used |= row_diffs(row)
        return rows

    return best_rows

# EVOLVE-BLOCK-END
