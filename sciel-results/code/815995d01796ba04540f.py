import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin vs 120s kill

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2))

    def validate(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    used.add(row[j] - ri)
        return used

    def scope_of(rows):
        return max(max(r) for r in rows)

    # ---- MRV interleaved construction (structural core) ----
    def build_interleaved(seed):
        random.seed(seed)
        rows = [[0] for _ in range(n_rows)]
        used = set()
        total_steps = n_rows * k
        steps = 0
        current_max = 0

        while steps < total_steps:
            if time.time() > deadline - 5:
                return None
            candidates = []
            for r in range(n_rows):
                if len(rows[r]) == k + 1:
                    continue
                cur = rows[r]
                last = cur[-1]
                valid_xs = []
                x = last + 1
                checks = 0
                # Tighter candidate window: up to 8 valid or 150 checks
                while len(valid_xs) < 8 and checks < 150:
                    if x in used:
                        x += 1
                        checks += 1
                        continue
                    conflict = False
                    for a in cur[1:]:
                        if (x - a) in used:
                            conflict = True
                            break
                    if not conflict:
                        valid_xs.append(x)
                    x += 1
                    checks += 1
                if not valid_xs:
                    return None
                candidates.append((valid_xs[0], r, valid_xs))

            if not candidates:
                break
            candidates.sort(key=lambda item: item[0])

            # Pick the most constrained (smallest feasible) with slight randomness
            if random.random() < 0.80:
                chosen_idx = 0
            else:
                chosen_idx = random.randint(0, min(4, len(candidates) - 1))
            _, r, valid_xs = candidates[chosen_idx]

            # Choose a value; mostly the tightest, occasionally explore
            if random.random() < 0.75:
                chosen_x = valid_xs[0]
            else:
                chosen_x = valid_xs[random.randint(0, min(4, len(valid_xs) - 1))]

            rows[r].append(chosen_x)
            for a in rows[r][:-1]:
                used.add(chosen_x - a)
            if chosen_x > current_max:
                current_max = chosen_x
            steps += 1

        if not validate(rows):
            return None
        return rows

    # ---- Epigraph local search with explicit scope variable t ----
    def local_search(rows, time_budget):
        rows = [r[:] for r in rows]
        diffs = compute_diffs(rows)
        t = scope_of(rows)  # auxiliary variable t = scope
        best_rows = [r[:] for r in rows]
        best_t = t
        end = time.time() + time_budget
        temp = max(1.0, (t - lower_bound) / 8.0)

        while time.time() < end:
            r = random.randint(0, n_rows - 1)
            row = rows[r]
            if len(row) < 2:
                continue
            idx = random.randint(1, len(row) - 1)
            # Wider delta window for better exploration
            deltas = list(range(-30, 31))
            random.shuffle(deltas)
            improved = False
            for delta in deltas:
                if delta == 0:
                    continue
                new_val = row[idx] + delta
                if new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
                old_diffs = set()
                new_diffs = set()
                for j in range(idx):
                    old_diffs.add(row[idx] - row[j])
                    new_diffs.add(new_val - row[j])
                for j in range(idx + 1, len(row)):
                    old_diffs.add(row[j] - row[idx])
                    new_diffs.add(row[j] - new_val)
                if new_diffs & (diffs - old_diffs):
                    continue
                new_t = max(t, new_val)
                new_diffs_set = (diffs - old_diffs) | new_diffs
                # Accept if t improves, or t equal with small probability
                if new_t < t or (new_t == t and random.random() < 0.15):
                    rows[r][idx] = new_val
                    diffs = new_diffs_set
                    t = new_t
                    if t < best_t:
                        best_t = t
                        best_rows = [rr[:] for rr in rows]
                    improved = True
                    break
            if not improved:
                temp *= 0.999

        return best_rows, best_t

    # ---- High-precision polish pass (tighter deltas, more iterations) ----
    def polish(rows, time_budget):
        rows = [r[:] for r in rows]
        diffs = compute_diffs(rows)
        t = scope_of(rows)
        best_rows = [r[:] for r in rows]
        best_t = t
        end = time.time() + time_budget

        while time.time() < end:
            r = random.randint(0, n_rows - 1)
            row = rows[r]
            if len(row) < 2:
                continue
            idx = random.randint(1, len(row) - 1)
            # Tighter delta window for precision
            deltas = list(range(-5, 6))
            random.shuffle(deltas)
            improved = False
            for delta in deltas:
                if delta == 0:
                    continue
                new_val = row[idx] + delta
                if new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
                old_diffs = set()
                new_diffs = set()
                for j in range(idx):
                    old_diffs.add(row[idx] - row[j])
                    new_diffs.add(new_val - row[j])
                for j in range(idx + 1, len(row)):
                    old_diffs.add(row[j] - row[idx])
                    new_diffs.add(row[j] - new_val)
                if new_diffs & (diffs - old_diffs):
                    continue
                new_t = max(t, new_val)
                if new_t < t:
                    new_diffs_set = (diffs - old_diffs) | new_diffs
                    rows[r][idx] = new_val
                    diffs = new_diffs_set
                    t = new_t
                    if t < best_t:
                        best_t = t
                        best_rows = [rr[:] for rr in rows]
                    improved = True
                    break
        return best_rows, best_t

    # ---- Fallback simple greedy (row-by-row) ----
    def greedy_fallback():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = {x - a for a in row}
                if not (nd & used):
                    row.append(x)
                    used |= nd
            rows.append(row)
        return rows

    best_rows = None
    best_scope = float('inf')

    # Phase 1: many interleaved MRV constructions
    seed = 1
    while time.time() < deadline - 30:
        rows = build_interleaved(seed)
        if rows is not None:
            s = scope_of(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows
        seed += 1

    if best_rows is None:
        best_rows = greedy_fallback()
        best_scope = scope_of(best_rows)

    # Phase 2: epigraph local search on the best found
    if time.time() < deadline - 10:
        ls_rows, ls_scope = local_search(best_rows, max(0.0, deadline - 10 - time.time()))
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope

    # Phase 3: high-precision polish pass
    if time.time() < deadline - 2:
        pl_rows, pl_scope = polish(best_rows, max(0.0, deadline - 2 - time.time()))
        if pl_scope < best_scope:
            best_rows = pl_rows
            best_scope = pl_scope

    return best_rows
