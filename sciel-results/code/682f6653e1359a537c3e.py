import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    # Total number of positive differences needed
    num_diffs = n_rows * k * (k + 1) // 2
    lb = num_diffs  # counting lower bound (scope must be at least this)

    best_rows = None
    best_scope = float('inf')

    # --- Fast greedy baseline (smallest valid x each step) ---
    def greedy_baseline():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # --- Backtracking search with bitset for speed ---
    def backtracking_search(max_scope, seed, time_limit=2.0):
        rng = random.Random(seed)
        max_diff = max_scope
        used_diffs = [False] * (max_diff + 1)
        rows = []
        k1 = k + 1
        t_start = time.time()

        def build_row(pos, last_val, row_so_far):
            nonlocal used_diffs
            if time.time() - t_start > time_limit:
                return None
            if pos == k1:
                return list(row_so_far)

            # Prune: need (k1 - pos) more elements, each >= 1 apart
            max_x = max_scope - (k1 - pos - 1)
            if max_x <= last_val:
                return None

            valid = []
            for x in range(last_val + 1, max_x + 1):
                ok = True
                for a in row_so_far:
                    if used_diffs[x - a]:
                        ok = False
                        break
                if ok:
                    valid.append(x)

            if not valid:
                return None

            # Bias toward smaller values, but shuffle a window
            n_take = min(10, len(valid))
            window = valid[:n_take]
            if len(valid) > n_take and rng.random() < 0.4:
                extra = rng.sample(valid[n_take:], min(5, len(valid) - n_take))
                window.extend(extra)
            rng.shuffle(window)

            for x in window:
                new_diffs = []
                for a in row_so_far:
                    d = x - a
                    new_diffs.append(d)
                    used_diffs[d] = True
                res = build_row(pos + 1, x, row_so_far + [x])
                if res is not None:
                    return res
                for d in new_diffs:
                    used_diffs[d] = False
            return None

        for _ in range(n_rows):
            row = build_row(1, 0, [0])
            if row is None:
                return None
            rows.append(row)
            for i in range(k1):
                for j in range(i + 1, k1):
                    used_diffs[row[j] - row[i]] = True
        return rows

    # --- Local improvement: shift elements left to reduce scope ---
    def local_improve(rows, seed, max_iters=2000):
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = max(max(r) for r in rows)
        k1 = k + 1

        # Build used set
        used = set()
        for row in rows:
            for i in range(k1):
                for j in range(i + 1, k1):
                    used.add(row[j] - row[i])

        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue

            # Remove element and its diffs
            for a in row[:elem_idx]:
                used.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used.discard(a - old_val)
            row.pop(elem_idx)

            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1
            found = False
            # Try from max_val down to min_val (prefer smallest valid)
            for new_val in range(min_val, max_val + 1):
                valid = True
                for a in row[:elem_idx]:
                    if (new_val - a) in used:
                        valid = False
                        break
                if valid:
                    for a in row[elem_idx:]:
                        if (a - new_val) in used:
                            valid = False
                            break
                if valid:
                    row.insert(elem_idx, new_val)
                    for a in row[:elem_idx]:
                        used.add(new_val - a)
                    for a in row[elem_idx + 1:]:
                        used.add(a - new_val)
                    found = True
                    break

            if found:
                new_scope = max(max(r) for r in rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used.add(a - old_val)

        return rows, scope

    # --- Systematic compaction pass ---
    def compact_pass(rows, seed=0):
        rows = [list(r) for r in rows]
        scope = max(max(r) for r in rows)
        k1 = k + 1
        used = set()
        for row in rows:
            for i in range(k1):
                for j in range(i + 1, k1):
                    used.add(row[j] - row[i])

        improved = True
        rounds = 0
        while improved and time.time() < deadline - 2 and rounds < 200:
            improved = False
            rounds += 1
            for row_idx in range(n_rows):
                row = rows[row_idx]
                for elem_pos in range(len(row) - 1, 0, -1):
                    old_val = row[elem_pos]
                    if old_val <= 1:
                        continue
                    for a in row[:elem_pos]:
                        used.discard(old_val - a)
                    for a in row[elem_pos + 1:]:
                        used.discard(a - old_val)
                    row.pop(elem_pos)
                    min_val = row[elem_pos - 1] + 1 if elem_pos > 0 else 1
                    found = False
                    for new_val in range(min_val, old_val):
                        valid = True
                        for a in row[:elem_pos]:
                            if (new_val - a) in used:
                                valid = False
                                break
                        if valid:
                            for a in row[elem_pos:]:
                                if (a - new_val) in used:
                                    valid = False
                                    break
                        if valid:
                            row.insert(elem_pos, new_val)
                            for a in row[:elem_pos]:
                                used.add(new_val - a)
                            for a in row[elem_pos + 1:]:
                                used.add(a - new_val)
                            found = True
                            break
                    if found:
                        new_scope = max(max(r) for r in rows)
                        if new_scope < scope:
                            scope = new_scope
                            improved = True
                    else:
                        row.insert(elem_pos, old_val)
                        for a in row[:elem_pos]:
                            used.add(old_val - a)
                        for a in row[elem_pos + 1:]:
                            used.add(a - old_val)
        return rows, scope

    # --- NEW: Difference-set based construction ---
    # Formulate as: partition the set D = {1, 2, ..., M} into n_rows subsets,
    # each of size k(k+1)/2, such that each subset can be realized as the
    # difference set of an increasing sequence of k+1 integers starting at 0.
    # We use a greedy + backtracking approach on the difference set.

    def can_realize_diffs(diff_set, max_scope, used_global, rng, time_limit):
        """Try to find a sequence of k+1 elements starting at 0 whose
        difference set is exactly diff_set, with all elements <= max_scope,
        and no difference colliding with used_global."""
        if time.time() > time_limit:
            return None

        k1 = k + 1
        # We need to build a sequence a_0=0 < a_1 < ... < a_{k-1}
        # such that {a_j - a_i : 0 <= i < j <= k-1} = diff_set
        # and all differences not in used_global.

        # Strategy: build the sequence element by element.
        # At each step, we pick the next element a_pos.
        # The new differences are {a_pos - a_i : i < pos}.
        # These must all be in diff_set and not in used_global.
        # After placing all elements, the total differences must equal diff_set.

        # This is complex. Instead, use a simpler heuristic:
        # Greedily build the sequence, picking the smallest valid next element.

        row = [0]
        remaining_diffs = set(diff_set)
        t_start = time.time()

        while len(row) < k1:
            pos = len(row)
            # Find smallest x > row[-1] such that {x - a for a in row}
            # is a subset of remaining_diffs and disjoint from used_global
            found = False
            for x in range(row[-1] + 1, max_scope + 1):
                if time.time() - t_start > time_limit:
                    return None
                new_diffs = {x - a for a in row}
                if new_diffs & used_global:
                    continue
                if not new_diffs.issubset(remaining_diffs):
                    continue
                # Check if we can still complete: remaining after this step
                # must have size (k1 - pos - 1) * (k1 - pos) // 2
                rem_after = remaining_diffs - new_diffs
                needed = (k1 - pos - 1) * (k1 - pos) // 2
                if len(rem_after) < needed:
                    continue
                # Tentatively place x
                row.append(x)
                remaining_diffs -= new_diffs
                found = True
                break

            if not found:
                return None

        # Verify exact match
        if remaining_diffs:
            return None

        return row

    def diff_set_construction(max_scope, seed, time_limit=3.0):
        """Construct n_rows rows by partitioning {1..max_scope} into difference sets."""
        rng = random.Random(seed)
        k1 = k + 1
        diffs_per_row = k * (k + 1) // 2
        t_start = time.time()

        # We need to assign each difference value 1..max_scope to at most one row.
        # Each row needs exactly diffs_per_row differences.
        # Total needed: n_rows * diffs_per_row
        # Available: max_scope
        # So max_scope >= n_rows * diffs_per_row

        if max_scope < n_rows * diffs_per_row:
            return None

        used_global = set()
        rows = []

        for r in range(n_rows):
            if time.time() - t_start > time_limit:
                return None

            # Pick a random subset of unused differences of size diffs_per_row
            available = [d for d in range(1, max_scope + 1) if d not in used_global]
            if len(available) < diffs_per_row:
                return None

            # Try multiple random subsets
            success = False
            for attempt in range(20):
                if time.time() - t_start > time_limit:
                    return None

                # Sample diffs_per_row values from available
                # Bias toward smaller values for tighter scope
                diff_subset = set(rng.sample(available, diffs_per_row))

                # Try to realize this difference set as a valid row
                row = can_realize_diffs(diff_subset, max_scope, used_global, rng,
                                         time_limit - (time.time() - t_start))
                if row is not None:
                    rows.append(row)
                    used_global |= diff_subset
                    success = True
                    break

            if not success:
                return None

        return rows

    # Phase 1: Greedy baseline
    rows = greedy_baseline()
    if rows:
        best_scope = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]

    # Phase 2: Local improvement on greedy
    if best_rows:
        improved, imp_scope = local_improve(best_rows, 42, max_iters=500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
        # Compact pass
        compacted, comp_scope = compact_pass(best_rows)
        if comp_scope < best_scope:
            best_scope = comp_scope
            best_rows = compacted

    # Phase 3: Backtracking search with decreasing target scope
    target = best_scope - 1
    attempt = 0
    while target > lb and time.time() < deadline - 10:
        found = False
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            sol = backtracking_search(target, 100000 + attempt, time_limit=1.5)
            if sol:
                new_scope = max(max(r) for r in sol)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in sol]
                    found = True
                    # Improve
                    improved, imp_scope = local_improve(best_rows, attempt + 500, max_iters=300)
                    if imp_scope < best_scope:
                        best_scope = imp_scope
                        best_rows = improved
                    compacted, comp_scope = compact_pass(best_rows)
                    if comp_scope < best_scope:
                        best_scope = comp_scope
                        best_rows = compacted
                    break
            attempt += 1
        if found:
            target = best_scope - 1
        else:
            target -= 1
            if target <= lb:
                break

    # Phase 4: More local improvement and compaction
    if best_rows and time.time() < deadline - 5:
        improved, imp_scope = local_improve(best_rows, 99999, max_iters=1000)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
        compacted, comp_scope = compact_pass(best_rows)
        if comp_scope < best_scope:
            best_scope = comp_scope
            best_rows = compacted

    # Phase 5: Difference-set based construction for tighter scopes
    if best_rows and time.time() < deadline - 8:
        target_ds = best_scope - 1
        while target_ds > lb and time.time() < deadline - 8:
            found_ds = False
            for i in range(15):
                if time.time() > deadline - 8:
                    break
                sol = diff_set_construction(target_ds, 200000 + i * 7919, time_limit=2.0)
                if sol:
                    new_scope = max(max(r) for r in sol)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in sol]
                        found_ds = True
                        # Improve
                        improved, imp_scope = local_improve(best_rows, 300000 + i, max_iters=400)
                        if imp_scope < best_scope:
                            best_scope = imp_scope
                            best_rows = improved
                        compacted, comp_scope = compact_pass(best_rows)
                        if comp_scope < best_scope:
                            best_scope = comp_scope
                            best_rows = compacted
                        break
            if found_ds:
                target_ds = best_scope - 1
            else:
                target_ds -= 1
                if target_ds <= lb:
                    break

    # Phase 6: Final random restarts with backtracking if time remains
    if best_rows and time.time() < deadline - 5:
        for i in range(30):
            if time.time() > deadline - 3:
                break
            sol = backtracking_search(best_scope - 1, 500000 + i * 131, time_limit=1.0)
            if sol:
                new_scope = max(max(r) for r in sol)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in sol]
                    improved, imp_scope = local_improve(best_rows, 700000 + i, max_iters=200)
                    if imp_scope < best_scope:
                        best_scope = imp_scope
                        best_rows = improved

    return best_rows


if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
