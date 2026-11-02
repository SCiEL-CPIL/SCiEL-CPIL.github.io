import time
import random
import math

_DEADLINE = 145.0  # safety margin under the 150 s wall-clock kill


def construct(n_rows, k):
    start = time.time()
    total_diffs_per_row = k * (k + 1) // 2  # C(k+1, 2)
    # counting lower bound on the largest entry
    lb = n_rows * total_diffs_per_row

    best_rows = None
    best_scope = None

    def elapsed():
        return time.time() - start

    def greedy_once(rng):
        """Greedy ruler construction; returns rows and scope."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        scope = max(r[-1] for r in rows)
        return rows, scope

    # Establish an initial upper bound with a few greedy tries (different random seeds
    # don't change greedy, so just one deterministic greedy gives the baseline).
    base_rows, base_scope = greedy_once(None)
    best_rows = base_rows
    best_scope = base_scope

    def backtrack(rows_so_far, used, current_scope_target, rng):
        """
        Try to build the remaining rows so that the overall scope is < current_scope_target.
        Returns a complete list of rows if found, else None.
        """
        if len(rows_so_far) == n_rows:
            return list(rows_so_far)
        if elapsed() > _DEADLINE:
            return None

        row = [0]
        used_row = set()
        x = 0

        # We want the final max entry of this row to be as small as possible,
        # but we must keep it under current_scope_target.
        max_allowed = current_scope_target - 1  # strictly better than best_scope

        def build_row():
            nonlocal row, used_row, x
            if len(row) == k + 1:
                # Row complete; commit
                rows_so_far.append(list(row))
                # Recurse for next row
                result = backtrack(rows_so_far, used, current_scope_target, rng)
                if result is not None:
                    return result
                rows_so_far.pop()
                return None

            # Try candidate marks from smallest upward, but bounded by max_allowed
            # We'll iterate x from 1..max_allowed
            while x <= max_allowed:
                x += 1
                new = {x - a for a in row}
                if new & used:
                    continue
                # commit
                row.append(x)
                used_row |= new
                res = build_row()
                if res is not None:
                    return res
                # undo
                row.pop()
                used_row -= new
            return None

        res = build_row()
        return res

    # Iteratively try to improve the scope by targeting best_scope - 1, then -2, ...
    # Use randomized restarts to vary the search (e.g., shuffle which candidate marks
    # are tried first within a bounded window).
    target = best_scope - 1
    restart = 0
    while target >= lb and elapsed() < _DEADLINE:
        found = False
        # A few randomized restarts per target to escape local traps
        for attempt in range(3):
            if elapsed() > _DEADLINE:
                break
            rng = random.Random(restart * 1000 + attempt)
            rows_found = backtrack([], set(), target + 1, rng)
            if rows_found is not None:
                # Verify validity and compute actual scope
                all_diffs = set()
                valid = True
                for r in rows_found:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            d = r[j] - r[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                if valid:
                    actual_scope = max(r[-1] for r in rows_found)
                    if actual_scope < best_scope:
                        best_scope = actual_scope
                        best_rows = rows_found
                        target = best_scope - 1
                        found = True
                        break
            restart += 1
        if found:
            continue
        # If we couldn't find a solution at this target, try a slightly larger target
        # by allowing a bit more slack (best_scope - 1 was the target; if it failed,
        # we can't do better than best_scope, so stop trying smaller targets).
        # Actually, if target = best_scope - 1 fails, no smaller scope is possible
        # given our search, so we stop.
        break

    return best_rows
