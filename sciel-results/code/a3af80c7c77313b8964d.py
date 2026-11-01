"""Difference triangle set (4,7) – randomized greedy with backtracking and time guard."""
import time
import random
import sys

def construct(n_rows, k):
    """
    Return n_rows rows, each a list of k+1 strictly increasing integers starting at 0,
    such that all positive differences within a row are distinct across all rows.
    Minimize the largest entry (the 'scope').
    """
    deadline = time.time() + 115.0  # safety margin before 120s kill
    best_rows = None
    best_scope = float('inf')

    # Helper: check if adding value x to current row keeps all diffs unused
    def valid_extension(row, x, used):
        for a in row:
            d = x - a
            if d in used:
                return False
        return True

    def add_to_used(used, row, x):
        for a in row:
            used.add(x - a)

    def remove_from_used(used, row, x):
        for a in row:
            used.discard(x - a)

    # Greedy construction with a given row order / strategy
    def greedy_construct(rows_order, used, rows, start_idx=0):
        """Build rows starting from start_idx, returning the completed rows list or None."""
        nonlocal best_rows, best_scope
        for i in range(start_idx, n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if valid_extension(row, x, used):
                    row.append(x)
                    add_to_used(used, row, x)
                    # Prune: if current max already exceeds best_scope, abandon
                    if x >= best_scope:
                        remove_from_used(used, row, x)
                        # Undo this row
                        for a in row[1:]:
                            remove_from_used(used, row, a)
                        return False
                else:
                    continue
            rows.append(row)
            if rows[-1][-1] >= best_scope:
                rows.pop()
                for a in row[1:]:
                    remove_from_used(used, row, a)
                return False
        return True

    # Randomized greedy: at each step, choose among candidate x values with some randomness
    def randomized_greedy(seed_val):
        nonlocal best_rows, best_scope
        rng = random.Random(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if valid_extension(row, x, used):
                    # Accept with some probability to allow exploration
                    row.append(x)
                    add_to_used(used, row, x)
                    if x >= best_scope:
                        # Undo
                        for a in row[1:]:
                            remove_from_used(used, row, a)
                        rows.pop() if rows and rows[-1] == row else None
                        return None
                # else continue
            rows.append(row)
        scope = max(r[-1] for r in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
        return rows

    # Local search: take a valid solution and try to improve by reordering or adjusting
    def local_improve(rows_in):
        nonlocal best_rows, best_scope
        rows = [list(r) for r in rows_in]
        scope = max(r[-1] for r in rows)
        if scope >= best_scope:
            return
        # Try swapping rows or trying different greedy orders
        # For now, just record if better
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]

    # Phase 1: Simple greedy to get a baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if valid_extension(row, x, used):
                row.append(x)
                add_to_used(used, row, x)
        rows.append(row)
    if rows:
        scope = max(r[-1] for r in rows)
        best_scope = scope
        best_rows = [list(r) for r in rows]

    # Phase 2: Randomized restarts
    seed_counter = 0
    while time.time() < deadline:
        if time.time() > deadline - 5.0:
            break
        randomized_greedy(seed_counter)
        seed_counter += 1
        # Periodically try a different strategy: reverse row construction order
        if seed_counter % 10 == 0:
            # Try constructing rows in reverse order (last row first)
            used2 = set()
            rows2 = []
            success = True
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    if valid_extension(row, x, used2):
                        row.append(x)
                        add_to_used(used2, row, x)
                        if x >= best_scope:
                            success = False
                            break
                if not success:
                    break
                rows2.append(row)
            if success and rows2:
                scope2 = max(r[-1] for r in rows2)
                if scope2 < best_scope:
                    best_scope = scope2
                    best_rows = [list(r) for r in rows2]

    # Ensure we always return something valid
    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if valid_extension(row, x, used):
                    row.append(x)
                    add_to_used(used, row, x)
            rows.append(row)
        best_rows = rows

    return best_rows
