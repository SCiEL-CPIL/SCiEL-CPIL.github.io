import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin from 150s limit

    MAX_SCOPE = 256  # bitmask size; best known is 112

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _all_diffs(rows):
        used = set()
        for row in rows:
            used |= _row_diffs(row)
        return used

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

    def _soft_min_cost(new_diffs, used_mask):
        """Smooth surrogate: soft-min of 1/d over new differences.
        Lower cost = uses larger (scarcer) differences = better fit.
        Using log-sum-exp style for numerical stability."""
        if not new_diffs:
            return 0.0
        # Sum of 1/d^2 (smoother than 1/d, less sensitive to tiny d)
        total = 0.0
        for d in new_diffs:
            if d >= MAX_SCOPE:
                return float('inf')
            inv = 1.0 / (d * d)
            total += inv
        return total

    def backtrack_solve(target_scope, seed, deadline, max_candidates_per_level=15):
        """Backtracking with bitmask and smooth cost ordering.
        Returns rows if a valid solution with scope < target_scope found, else None."""
        rng = random.Random(seed)
        used_mask = 0
        rows = []

        def _time_up():
            return time.time() > deadline

        def _backtrack(row_idx, current_row):
            nonlocal used_mask, rows

            if _time_up():
                return False

            if row_idx == n_rows:
                # All rows done
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < target_scope:
                    return True
                return True  # scope <= target_scope - 1 by construction

            if len(current_row) == k + 1:
                rows.append(current_row)
                if _backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False

            last_val = current_row[-1]
            upper = target_scope - 1  # x must be < target_scope

            candidates = []
            for x in range(last_val + 1, upper + 1):
                # Check conflicts via bitmask
                conflict = False
                new_mask = 0
                for a in current_row:
                    d = x - a
                    if d >= MAX_SCOPE:
                        conflict = True
                        break
                    bit = 1 << d
                    if used_mask & bit:
                        conflict = True
                        break
                    new_mask |= bit
                if conflict:
                    continue

                # Smooth cost: soft-min surrogate
                new_diffs = {x - a for a in current_row}
                cost = _soft_min_cost(new_diffs, used_mask)
                # Prefer smaller x for compactness
                cost += x * 0.005
                # Small noise for exploration
                cost += rng.random() * 0.01

                candidates.append((cost, x, new_mask))

                if len(candidates) >= max_candidates_per_level:
                    break

            if not candidates:
                return False

            candidates.sort(key=lambda t: t[0])

            # Shuffle top few for exploration
            top_n = min(5, len(candidates))
            top = candidates[:top_n]
            rng.shuffle(top)
            rest = candidates[top_n:]

            for cost, x, new_mask in top + rest:
                old_mask = used_mask
                used_mask |= new_mask
                current_row.append(x)

                if _backtrack(row_idx, current_row):
                    return True

                current_row.pop()
                used_mask = old_mask

            return False

        if _backtrack(0, [0]):
            return [r[:] for r in rows]
        return None

    def local_search_shrink(rows, deadline, rng):
        """Reduce max entry by trying to lower individual elements."""
        rows = [list(r) for r in rows]
        used = _all_diffs(rows)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_scope = _scope(rows)

            for ri in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[ri]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    if old_val < current_scope:
                        continue

                    low = row[pos - 1] + 1
                    high = old_val - 1
                    if pos + 1 < len(row):
                        high = min(high, row[pos + 1] - 1)

                    if low > high:
                        continue

                    candidates = list(range(high, low - 1, -1))
                    rng.shuffle(candidates)

                    for new_val in candidates:
                        if time.time() > deadline:
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

                        conflict = new_diffs & (used - old_diffs)
                        if conflict:
                            continue

                        row[pos] = new_val
                        used -= old_diffs
                        used |= new_diffs
                        improved = True
                        break

        return rows

    def greedy_construct_bestfit(seed_offset=0, max_candidates=30):
        """Fast greedy with smooth cost ordering for initial solution."""
        rng = random.Random(42 + seed_offset)
        used = set()
        rows = []

        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                candidates = []
                x = last + 1
                checked = 0
                while checked < max_candidates:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        cost = _soft_min_cost(new_diffs, used)
                        cost += x * 0.005
                        candidates.append((cost, x))
                    x += 1
                    checked += 1

                if not candidates:
                    x = last + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append((0, x))
                            break
                        x += 1

                candidates.sort(key=lambda t: t[0])
                best_x = candidates[0][1]

                if len(candidates) > 1 and rng.random() < 0.25:
                    idx = rng.randint(0, min(2, len(candidates) - 1))
                    best_x = candidates[idx][1]

                row.append(best_x)
                new_diffs = {best_x - a for a in row[:-1]}
                used |= new_diffs
            rows.append(row)

        return rows

    # --- Phase 1: Fast greedy to get an initial solution ---
    best_rows = greedy_construct_bestfit(seed_offset=0)
    best_scope = _scope(best_rows)

    for seed in range(10):
        if time.time() > deadline - 25:
            break
        rows = greedy_construct_bestfit(seed_offset=seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope

    # --- Phase 2: Local search polish ---
    rng = random.Random(123)
    if time.time() < deadline - 15:
        polished = local_search_shrink(best_rows, deadline - 10, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope

    # --- Phase 3: Progressive backtracking to reduce scope ---
    current_target = best_scope
    lower_bound = max(n_rows * k * (k + 1) // 2, 50)
    seed_counter = 0

    while time.time() < deadline - 10:
        target = current_target - 1
        if target < lower_bound:
            break

        # Try multiple seeds at this target
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            result = backtrack_solve(target, seed_counter, deadline - 10)
            seed_counter += 1
            if result is not None:
                result_scope = _scope(result)
                if result_scope < best_scope:
                    best_rows = result
                    best_scope = result_scope
                    current_target = best_scope
                found = True
                break

        if not found:
            current_target = target

        # Polish after each improvement
        if time.time() < deadline - 10:
            rng2 = random.Random(seed_counter)
            polished = local_search_shrink(best_rows, deadline - 10, rng2)
            polished_scope = _scope(polished)
            if polished_scope < best_scope:
                best_rows = polished
                best_scope = polished_scope
                current_target = best_scope

    # --- Phase 4: Final polish ---
    if time.time() < deadline - 3:
        rng3 = random.Random(999)
        polished = local_search_shrink(best_rows, deadline - 3, rng3)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope

    return best_rows
