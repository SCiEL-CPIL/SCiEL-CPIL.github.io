import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 115.0
    best_max = float('inf')
    best_rows = None

    # Number of pairwise differences per row: C(k+1, 2)
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # --- Helper Functions ---

    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def get_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def compute_max(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        return len(all_diffs) == len(set(all_diffs))

    # Given a set of diffs (size k*(k+1)/2), try to decompose into a valid row.
    # A valid row [0, a1, ..., ak] has diffs = {a_j - a_i : 0 <= i < j <= k}.
    # The largest diff is a_k. The diffs are closed under certain structure.
    # We use a greedy approach: sort diffs, the largest must be a_k.
    # Then a_k - d must also be in the set for each d that is a diff from 0.
    # Actually, the diffs from 0 are a_1, a_2, ..., a_k.
    # Strategy: pick a_k = max(diffs). Then the diffs from 0 are a subset S of diffs
    # such that S contains a_k, and for each pair in S, their difference is also in diffs.
    # This is essentially finding a "difference basis" of size k.

    def decompose_to_row(diff_set):
        """Try to find a row [0, a1, ..., ak] whose pairwise diffs equal diff_set."""
        if len(diff_set) != diffs_per_row:
            return None
        a_k = max(diff_set)
        # The diffs from 0 to each element: {a_1, ..., a_k}
        # These are k elements of diff_set, including a_k.
        # For any two elements x, y in this set, |x - y| must be in diff_set.
        # We search for a subset S of diff_set of size k containing a_k
        # that is "difference-closed" within diff_set.

        # Greedy: start with S = {a_k}, try to add elements.
        # An element d can be added if for all existing s in S, |d - s| is in diff_set
        # and (d - s) or (s - d) is in diff_set.
        # Actually we need: for all pairs in S, their absolute difference is in diff_set.

        # Try a heuristic: the diffs from 0 should be "spread out".
        # Let's try: S must contain a_k. Pick next largest that works, etc.
        # More systematic: try all subsets? Too expensive.
        # Heuristic: sort diff_set descending, greedily build S.

        candidates = sorted(diff_set, reverse=True)
        # a_k must be in S
        S = [a_k]
        remaining = [d for d in candidates if d != a_k]

        for d in remaining:
            if len(S) == k:
                break
            # Check if d can be added to S
            can_add = True
            for s in S:
                diff = abs(d - s)
                if diff == 0:
                    continue
                if diff not in diff_set:
                    can_add = False
                    break
            if can_add:
                S.append(d)

        if len(S) != k:
            return None

        # S should be {a_1, ..., a_k}. Sort ascending.
        S.sort()
        # Verify: pairwise diffs of S (including 0) should equal diff_set
        # Row is [0] + S
        row = [0] + S
        row_diffs = get_diffs_set(row)
        if row_diffs == diff_set:
            return row
        return None

    # --- Greedy Baseline ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = compute_max(best_rows)

    # --- Difference-set based construction ---
    # Choose n_rows disjoint subsets of size diffs_per_row from [1..M],
    # then decompose each into a row. Optimize M (scope).

    def diffset_construct(M, n_restarts=30, time_budget=10.0):
        """Try to construct rows using diff sets from [1..M]."""
        start_time = time.time()
        for restart in range(n_restarts):
            if time.time() - start_time > time_budget:
                break
            random.seed(12345 + restart * 7919)
            used = set()
            rows = []
            success = True
            for r in range(n_rows):
                # Pick diffs_per_row unused values from [1..M]
                available = [d for d in range(1, M + 1) if d not in used]
                if len(available) < diffs_per_row:
                    success = False
                    break
                # Random subset, biased toward larger values for spread
                # Actually, we want the subset to be decomposable.
                # Try: pick a_k first (largest), then build around it.
                # Simpler: random subset, try to decompose.
                # Multiple attempts per row
                found_row = None
                for attempt in range(50):
                    subset = set(random.sample(available, diffs_per_row))
                    row = decompose_to_row(subset)
                    if row is not None:
                        found_row = row
                        break
                if found_row is None:
                    success = False
                    break
                used |= get_diffs_set(found_row)
                rows.append(found_row)
            if success and len(rows) == n_rows:
                return rows, compute_max(rows)
        return None, None

    # --- Polish Phase: try to reduce the maximum entry ---
    def polish(rows, time_limit):
        rows = [r[:] for r in rows]
        current_max = compute_max(rows)
        start_time = time.time()

        diff_counts = defaultdict(int)
        for r in rows:
            for d in get_diffs(r):
                diff_counts[d] += 1

        # Check if valid
        for d, c in diff_counts.items():
            if c > 1:
                return rows, current_max

        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try to reduce the last element of each row
            for r_idx in range(n_rows):
                row = rows[r_idx]
                old_last = row[-1]
                # Try values from old_last-1 down to row[-2]+1
                for new_val in range(old_last - 1, row[-2], -1):
                    old_diffs_to_remove = [old_last - x for x in row[:-1]]
                    new_diffs_to_add = [new_val - x for x in row[:-1]]

                    # Check if new diffs are all unused
                    valid = True
                    for d in new_diffs_to_add:
                        if d in diff_counts:
                            valid = False
                            break

                    if valid:
                        # Remove old diffs
                        for d in old_diffs_to_remove:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        # Add new diffs
                        for d in new_diffs_to_add:
                            diff_counts[d] = 1
                        row[-1] = new_val
                        improved = True
                        current_max = compute_max(rows)
                        break

        return rows, current_max

    # --- Local Search: swap-based difference minimization ---
    def local_search(rows, time_limit, seed_offset=0):
        """
        Discrete gradient descent on the sorted difference vector.
        Each step: pick the largest used difference d, try to replace it with
        a smaller unused difference d' by adjusting the row containing d.
        """
        random.seed(seed_offset)
        rows = [r[:] for r in rows]
        n = n_rows
        L = row_len
        start_time = time.time()

        # Build diff tracking
        diff_to_rows = defaultdict(list)  # diff -> list of (row_idx, pair)
        row_diffs = [get_diffs(r) for r in rows]
        used_diffs = set()
        for rd in row_diffs:
            for d in rd:
                used_diffs.add(d)

        current_max = compute_max(rows)

        def rebuild_diff_tracking():
            nonlocal diff_to_rows, used_diffs
            diff_to_rows = defaultdict(list)
            used_diffs = set()
            for r_idx, row in enumerate(rows):
                for i in range(L):
                    for j in range(i + 1, L):
                        d = row[j] - row[i]
                        diff_to_rows[d].append((r_idx, i, j))
                        used_diffs.add(d)

        rebuild_diff_tracking()

        def try_reduce_row_last(r_idx, new_last):
            """Try to change row[r_idx][-1] to new_last. Returns True if successful."""
            row = rows[r_idx]
            old_last = row[-1]
            if new_last <= row[-2] or new_last >= old_last:
                return False
            # Check new diffs
            for i in range(L - 1):
                d = new_last - row[i]
                if d in used_diffs:
                    return False
            # Apply
            for i in range(L - 1):
                d_old = old_last - row[i]
                d_new = new_last - row[i]
                lst = diff_to_rows[d_old]
                lst.remove((r_idx, i, L - 1))
                if not lst:
                    del diff_to_rows[d_old]
                    used_diffs.discard(d_old)
                diff_to_rows[d_new].append((r_idx, i, L - 1))
                used_diffs.add(d_new)
            row[-1] = new_last
            return True

        def try_swap_entry(r_idx, i_idx, new_val):
            """Try to change row[r_idx][i_idx] to new_val. Returns True if successful."""
            row = rows[r_idx]
            old_val = row[i_idx]
            if new_val == old_val:
                return False
            # Check bounds
            if i_idx > 0 and new_val <= row[i_idx - 1]:
                return False
            if i_idx < L - 1 and new_val >= row[i_idx + 1]:
                return False
            # Check new diffs
            for j in range(L):
                if j == i_idx:
                    continue
                if j < i_idx:
                    d = new_val - row[j]
                else:
                    d = row[j] - new_val
                if d in used_diffs:
                    return False
            # Apply: remove old diffs involving i_idx
            for j in range(L):
                if j == i_idx:
                    continue
                if j < i_idx:
                    d_old = old_val - row[j]
                else:
                    d_old = row[j] - old_val
                lst = diff_to_rows[d_old]
                lst.remove((r_idx, min(i_idx, j), max(i_idx, j)))
                if not lst:
                    del diff_to_rows[d_old]
                    used_diffs.discard(d_old)
            # Add new diffs
            for j in range(L):
                if j == i_idx:
                    continue
                if j < i_idx:
                    d_new = new_val - row[j]
                else:
                    d_new = row[j] - new_val
                diff_to_rows[d_new].append((r_idx, min(i_idx, j), max(i_idx, j)))
                used_diffs.add(d_new)
            row[i_idx] = new_val
            return True

        best_max = current_max
        best_rows = [r[:] for r in rows]
        no_improve_count = 0
        max_no_improve = 20000

        while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
            # Pick a random row and position to try to reduce
            r_idx = random.randint(0, n - 1)
            i_idx = random.randint(1, L - 1)

            row = rows[r_idx]
            old_val = row[i_idx]
            lower = row[i_idx - 1] + 1
            if i_idx < L - 1:
                upper = row[i_idx + 1] - 1
            else:
                upper = old_val - 1

            if lower > upper:
                no_improve_count += 1
                continue

            # Try to find a value that reduces the max difference
            # Prefer values that are smaller than old_val
            new_val = random.randint(lower, upper)
            if new_val == old_val:
                no_improve_count += 1
                continue

            if try_swap_entry(r_idx, i_idx, new_val):
                new_max = compute_max(rows)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # Revert
                    try_swap_entry(r_idx, i_idx, old_val)
            else:
                no_improve_count += 1

        return best_rows, best_max

    # --- Main Loop ---
    remaining = deadline - time.time()

    # Phase 1: Try diffset construction with various M values
    # The counting bound is 105, best known is 112.
    # Try M from 105 upward.
    diffset_budget = min(remaining * 0.3, 30.0)
    if diffset_budget > 3.0:
        for M in range(105, 120):
            if time.time() > deadline - 5.0:
                break
            ds_rows, ds_max = diffset_construct(M, n_restarts=20, time_budget=diffset_budget / 15.0)
            if ds_rows is not None and is_valid(ds_rows):
                if ds_max < best_max:
                    best_rows = ds_rows
                    best_max = ds_max
                break  # Found a valid construction, stop increasing M

    # Phase 2: Local search to reduce scope
    ls_remaining = deadline - time.time() - 5.0
    if ls_remaining > 3.0:
        n_starts = 3
        time_per_start = ls_remaining / n_starts

        for s in range(n_starts):
            if time.time() >= deadline - 5.0:
                break

            if s == 0:
                start_rows = [r[:] for r in best_rows]
            else:
                # Perturb
                start_rows = [r[:] for r in best_rows]
                for _ in range(5):
                    r_idx = random.randint(0, n_rows - 1)
                    i_idx = random.randint(1, row_len - 1)
                    old_val = start_rows[r_idx][i_idx]
                    lower = start_rows[r_idx][i_idx - 1] + 1
                    if i_idx < row_len - 1:
                        upper = start_rows[r_idx][i_idx + 1] - 1
                    else:
                        upper = compute_max(start_rows) + 5
                    if lower <= upper:
                        start_rows[r_idx][i_idx] = random.randint(lower, upper)

            t_remaining = deadline - time.time() - 5.0
            if t_remaining <= 0:
                break

            ls_rows, ls_max = local_search(start_rows, t_remaining, seed_offset=777 + s * 313)
            if is_valid(ls_rows):
                if ls_max < best_max:
                    best_rows = ls_rows
                    best_max = ls_max

    # Phase 3: Final polish
    if time.time() < deadline - 1.0:
        polish_time = deadline - time.time() - 1.0
        polished_rows, polished_max = polish(best_rows, polish_time)
        if is_valid(polished_rows) and polished_max < best_max:
            best_rows = polished_rows
            best_max = polished_max

    return best_rows

    # --- Polish Phase: try to reduce the maximum entry ---
    def polish(rows, time_limit):
        rows = [r[:] for r in rows]
        current_max = compute_max(rows)
        start_time = time.time()

        diff_counts = defaultdict(int)
        for r in rows:
            for d in get_diffs(r):
                diff_counts[d] += 1

        # Check if valid
        for d, c in diff_counts.items():
            if c > 1:
                return rows, current_max

        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try to reduce the last element of each row
            for r_idx in range(n_rows):
                row = rows[r_idx]
                old_last = row[-1]
                # Try values from old_last-1 down to row[-2]+1
                for new_val in range(old_last - 1, row[-2], -1):
                    old_diffs_to_remove = [old_last - x for x in row[:-1]]
                    new_diffs_to_add = [new_val - x for x in row[:-1]]

                    # Check if new diffs are all unused
                    valid = True
                    for d in new_diffs_to_add:
                        if d in diff_counts:
                            valid = False
                            break

                    if valid:
                        # Remove old diffs
                        for d in old_diffs_to_remove:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        # Add new diffs
                        for d in new_diffs_to_add:
                            diff_counts[d] = 1
                        row[-1] = new_val
                        improved = True
                        current_max = compute_max(rows)
                        break

        return rows, current_max

    # --- Main Loop ---
    remaining = deadline - time.time()

    # First, try CSP solver with a time budget
    csp_budget = min(remaining * 0.4, 40.0)
    if csp_budget > 5.0:
        csp_start = time.time()
        csp_rows, csp_max = csp_solve()
        if csp_rows is not None and is_valid(csp_rows):
            if csp_max < best_max:
                best_rows = csp_rows
                best_max = csp_max

    return best_rows
