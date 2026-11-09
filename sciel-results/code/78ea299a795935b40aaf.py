"""Combined solver for difference triangle set (n_rows, k).
Uses backtracking search with candidate ordering and randomized restarts,
plus local-search refinement to reduce the scope."""
import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 130.0  # safety margin before 150s kill

    best_rows = None
    best_scope = float('inf')

    # --- Validation and scope helpers ---
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_scope(rows):
        return max(max(row) for row in rows)

    # --- Backtracking search: build one row given the set of already-used differences ---
    def backtrack_row(row_so_far, used_diffs, remaining, max_val):
        """Recursively build the rest of the row. Returns list or None."""
        if len(row_so_far) == k + 1:
            return list(row_so_far)
        if time.time() > deadline - 5:
            return None

        last = row_so_far[-1]
        # Try candidate values from last+1 upward, but cap at max_val
        for x in range(last + 1, max_val + 1):
            new_diffs = {x - a for a in row_so_far}
            if new_diffs & used_diffs:
                continue
            row_so_far.append(x)
            used_diffs |= new_diffs
            result = backtrack_row(row_so_far, used_diffs, remaining - 1, max_val)
            row_so_far.pop()
            used_diffs -= new_diffs
            if result is not None:
                return result
        return None

    # --- Full backtracking: build all rows with a given max scope ---
    def solve_with_max_scope(max_scope, rng):
        """Try to build n_rows rows all with entries <= max_scope."""
        used_diffs = set()
        rows = []

        def build_row(idx):
            if idx == n_rows:
                return True
            if time.time() > deadline - 5:
                return False

            # Generate candidate starting sequences: row must start at 0
            row = [0]
            # Use backtracking to fill the rest
            result = backtrack_row(row, used_diffs, k, max_scope)
            if result is None:
                return False
            rows.append(result)
            # Add all differences of this row to used_diffs
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    used_diffs.add(result[j] - result[i])

            if build_row(idx + 1):
                return True

            # Undo
            rows.pop()
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    used_diffs.discard(result[j] - result[i])
            return False

        if build_row(0):
            return rows
        return None

    # --- Greedy fallback (fast, always valid) ---
    def greedy_fallback():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # --- Local search: try to reduce scope by adjusting the largest entry ---
    def local_search(rows, deadline):
        nonlocal best_scope, best_rows
        best = [list(r) for r in rows]
        cur_scope = compute_scope(best)

        # Try reducing the last element of each row
        for _ in range(50):  # limited iterations
            if time.time() > deadline - 3:
                break
            improved = False
            for ri in range(n_rows):
                row = best[ri]
                if len(row) < 2:
                    continue
                last_val = row[-1]
                # Compute used diffs excluding those involving last element
                used_others = set()
                for oi in range(n_rows):
                    if oi == ri:
                        continue
                    orow = best[oi]
                    for i in range(len(orow)):
                        for j in range(i + 1, len(orow)):
                            used_others.add(orow[j] - orow[i])
                used_row_no_last = set()
                for i in range(len(row) - 1):
                    for j in range(i + 1, len(row) - 1):
                        used_row_no_last.add(row[j] - row[i])

                # Try to lower last_val
                for new_val in range(last_val - 1, row[-2], -1):
                    new_diffs = {new_val - a for a in row[:-1]}
                    if new_diffs & (used_others | used_row_no_last):
                        continue
                    if new_val <= row[-2]:
                        continue
                    # Tentative change
                    old_val = row[-1]
                    row[-1] = new_val
                    if is_valid(best):
                        ns = compute_scope(best)
                        if ns < cur_scope:
                            cur_scope = ns
                            improved = True
                            break
                    else:
                        row[-1] = old_val
                        break

            if not improved:
                break

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [list(r) for r in best]
        return best, cur_scope

    # --- Phase 1: Greedy to get a baseline ---
    greedy_rows = greedy_fallback()
    best_rows = greedy_rows
    best_scope = compute_scope(greedy_rows)

    # --- Phase 2: Try backtracking with decreasing max_scope ---
    # Start from best_scope - 1 and go down
    for max_scope in range(best_scope - 1, max(best_scope - 50, 50), -1):
        if time.time() > deadline - 10:
            break
        # Try a few randomized orderings via different RNG seeds
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            rng = random.Random(seed)
            result = solve_with_max_scope(max_scope, rng)
            if result is not None:
                scope = compute_scope(result)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = result
                    # Local search refinement
                    best_rows, best_scope = local_search(best_rows, deadline)
                break  # found a solution at this scope, try next lower

    # --- Phase 3: Additional randomized greedy with windows ---
    if time.time() < deadline - 15:
        for window in [2, 3, 5, 8]:
            if time.time() > deadline - 10:
                break
            for seed in range(20):
                if time.time() > deadline - 10:
                    break
                rng = random.Random(seed)
                used = set()
                rows = []
                success = True
                for _ in range(n_rows):
                    row = [0]
                    x = 0
                    while len(row) < k + 1:
                        found = False
                        for offset in range(window):
                            candidate = x + offset
                            if candidate <= 0:
                                continue
                            new_diffs = {candidate - a for a in row}
                            if not new_diffs & used:
                                row.append(candidate)
                                used |= new_diffs
                                x = candidate
                                found = True
                                break
                        if not found:
                            x += 1
                    rows.append(row)
                if is_valid(rows):
                    scope = compute_scope(rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = rows
                        if time.time() < deadline - 10:
                            best_rows, best_scope = local_search(best_rows, deadline)

    # --- Final local search pass ---
    if best_rows is not None and time.time() < deadline - 5:
        best_rows, best_scope = local_search(best_rows, deadline)

    # Safety: ensure we return something valid
    if best_rows is None or not is_valid(best_rows):
        best_rows = greedy_fallback()

    return best_rows
