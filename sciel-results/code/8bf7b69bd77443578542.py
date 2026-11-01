import time
import sys

def construct(n_rows, k):
    deadline = time.time() + 115.0
    total_diffs = n_rows * (k * (k + 1) // 2)
    lower_bound = total_diffs - 1  # m must be >= number of distinct diffs

    def greedy_rows(upper):
        # Build rows greedily, never exceeding `upper`. Returns list of rows or None.
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if x > upper:
                    return None
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Get an initial valid solution via greedy (scope = max entry).
    base = greedy_rows(10**9)
    if base is None:
        # Fallback: plain greedy without cap (shouldn't happen)
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = base
    best_scope = max(max(r) for r in base)

    # Try to improve by backtracking for smaller scopes.
    # Search scopes from lower_bound+1 up to best_scope-1, but prefer a few
    # promising ones. We'll do a bounded backtracking for a target scope.
    def backtrack(target_scope):
        # Returns a valid set of rows with max entry <= target_scope, or None.
        used = set()
        rows = []
        # MRV: pick the row that currently has the fewest feasible next marks.
        # We build rows one by one; for each row choose the next mark.
        # To keep it simple & effective: build rows in order, but choose the
        # next mark for the current row by MRV (fewest options given used).
        # Since rows are independent except via `used`, order of building rows
        # matters; we build the "hardest" row first. Approximate difficulty by
        # trying to build each row and seeing how far it gets.
        # Simpler robust approach: build rows in a fixed order but with
        # backtracking over the choice of marks, using MRV within a row.

        # We'll implement a recursive search over rows.
        # State: rows built so far, used set.
        # For the next row, we must fill k+1 marks starting at 0.
        # We choose the mark sequence with backtracking.

        # To keep runtime bounded, limit recursion depth via a node counter.
        node_counter = [0]
        max_nodes = 200000

        def build_row(start_index, current_row, used_set):
            # Fill current_row from len(current_row) to k+1.
            if time.time() > deadline:
                return None
            node_counter[0] += 1
            if node_counter[0] > max_nodes:
                return None
            if len(current_row) == k + 1:
                return current_row
            last = current_row[-1]
            # Candidate next marks: last+1 .. target_scope
            # MRV: choose the candidate that leaves the most future feasibility.
            # Heuristic: try candidates that minimize conflicts; we just try in
            # increasing order but prune early.
            for x in range(last + 1, target_scope + 1):
                new = {x - a for a in current_row}
                if new & used_set:
                    continue
                # Tentatively place
                current_row.append(x)
                used_set |= new
                result = build_row(start_index, current_row, used_set)
                if result is not None:
                    return result
                # Undo
                used_set -= new
                current_row.pop()
            return None

        def build_rows(idx, used_set, rows_so_far):
            if time.time() > deadline:
                return None
            if idx == n_rows:
                return rows_so_far
            # MRV over which row to build next: estimate difficulty.
            # We'll just build in order but allow trying different rows.
            # For simplicity build in fixed order (row idx).
            row = [0]
            result = build_row(idx, row, used_set)
            if result is None:
                return None
            used_set |= {result[j] - result[i] for i in range(len(result)) for j in range(i+1, len(result))}
            return build_rows(idx + 1, used_set, rows_so_far + [result])

        return build_rows(0, set(), [])

    # Try a handful of target scopes, from a bit below best down toward lower_bound.
    # Limit number of attempts to stay within time.
    attempts = 0
    max_attempts = 12
    # Candidate scopes: best_scope-1, best_scope-2, ..., and some near lower_bound.
    candidates = list(range(best_scope - 1, max(lower_bound, best_scope - 40) - 1, -1))
    # Also add a few near the lower bound for a jump.
    for s in range(lower_bound + 1, min(lower_bound + 15, best_scope - 1) + 1):
        if s not in candidates:
            candidates.append(s)
    # Sort candidates so we try smaller scopes first (bigger improvement),
    # but only if we have time; otherwise larger scopes are easier.
    # Compromise: try a mix. We'll try in descending order of difficulty of
    # finding, i.e., start a bit below best and step down.
    candidates.sort()
    for scope in candidates:
        if scope >= best_scope:
            continue
        if time.time() > deadline - 5:
            break
        if attempts >= max_attempts:
            break
        attempts += 1
        res = backtrack(scope)
        if res is not None:
            sc = max(max(r) for r in res)
            if sc < best_scope:
                best_scope = sc
                best_rows = res
                # Stop if we hit the lower bound.
                if best_scope <= lower_bound:
                    break

    return best_rows
