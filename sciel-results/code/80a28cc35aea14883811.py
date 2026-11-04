"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math

_DEADLINE = None
_TIME_LIMIT = 140.0


def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def construct(n_rows, k):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    total_diffs = n_rows * k * (k + 1) // 2

    # Counting lower bound
    lower_bound = 0
    s = 0
    while s < total_diffs:
        lower_bound += 1
        s += lower_bound

    rng = random.Random(42)

    # --- Bitset-based used-difference tracking for speed ---
    def make_bitset(scope):
        return bytearray(scope + 2)

    def check_diffs(used_bits, row, new_val, scope):
        """Check if adding new_val to row creates conflicts. Returns True if OK."""
        for a in row:
            d = new_val - a
            if d > scope:
                return False
            if used_bits[d]:
                return False
        return True

    def mark_diffs(used_bits, row, new_val):
        """Mark differences created by adding new_val to row."""
        for a in row:
            d = new_val - a
            used_bits[d] = 1

    def unmark_diffs(used_bits, row, new_val):
        """Unmark differences created by adding new_val to row."""
        for a in row:
            d = new_val - a
            used_bits[d] = 0

    # --- Deterministic greedy baseline ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_max = max(max(r) for r in best_rows)

    # --- Simulated annealing with bitset for fast collision checks ---
    def simulated_annealing(initial_rows, target_scope, max_iters=5000):
        """
        Simulated annealing to minimize the max entry.
        Uses bitset for fast collision checks.
        """
        rows = [list(r) for r in initial_rows]
        scope = target_scope
        used_bits = make_bitset(scope)

        # Initialize used_bits
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= scope:
                        used_bits[d] = 1

        cur_max = max(max(r) for r in rows)
        best_rows_sa = [list(r) for r in rows]
        best_max_sa = cur_max

        T = 10.0
        cooling = 0.9995

        for it in range(max_iters):
            if _time_left() < 5.0:
                break

            # Pick a random row and a random position to modify
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]

            # Pick a random element (not the first) to move
            pos = rng.randint(1, len(row) - 1)
            old_val = row[pos]

            # Unmark old differences
            unmark_diffs(used_bits, row, old_val)

            # Try to find a new valid position
            # Bias towards smaller values to reduce scope
            new_val = None
            search_range = max(1, old_val - rng.randint(0, 5))
            for candidate in range(row[pos - 1] + 1, min(old_val, scope) + 1):
                if check_diffs(used_bits, row, candidate, scope):
                    new_val = candidate
                    break

            if new_val is None:
                # Try a random position in a wider range
                for _ in range(10):
                    candidate = rng.randint(row[pos - 1] + 1, min(old_val + 3, scope))
                    if candidate > row[pos - 1] and check_diffs(used_bits, row, candidate, scope):
                        # Also need to ensure it doesn't conflict with subsequent elements
                        valid = True
                        for j in range(pos + 1, len(row)):
                            d = row[j] - candidate
                            if d <= 0 or used_bits[d]:
                                valid = False
                                break
                        if valid:
                            new_val = candidate
                            break

            if new_val is None:
                # Revert
                mark_diffs(used_bits, row, old_val)
                continue

            # Check that subsequent elements still work
            valid = True
            for j in range(pos + 1, len(row)):
                d = row[j] - new_val
                if d <= 0 or used_bits[d]:
                    valid = False
                    break

            if not valid:
                # Revert
                mark_diffs(used_bits, row, old_val)
                continue

            # Accept the move
            row[pos] = new_val
            mark_diffs(used_bits, row, new_val)

            # Recalculate max
            cur_max = max(max(r) for r in rows)
            if cur_max < best_max_sa:
                best_max_sa = cur_max
                best_rows_sa = [list(r) for r in rows]

            T *= cooling

        return best_rows_sa, best_max_sa

    # --- Local search with bitset ---
    def local_search_bitset(rows, target_scope, max_iters=3000):
        """
        Local search: try to reduce each element to the smallest valid value.
        Uses bitset for fast collision checks.
        """
        rows = [list(r) for r in rows]
        scope = target_scope
        used_bits = make_bitset(scope)

        # Initialize used_bits
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= scope:
                        used_bits[d] = 1

        improved = False
        for _ in range(max_iters):
            if _time_left() < 5.0:
                break

            any_progress = False
            for row_idx in range(n_rows):
                row = rows[row_idx]
                for pos in range(1, len(row)):
                    old_val = row[pos]
                    if old_val <= row[pos - 1] + 1:
                        continue

                    # Try to reduce this element
                    unmark_diffs(used_bits, row, old_val)

                    new_val = None
                    for candidate in range(row[pos - 1] + 1, old_val):
                        if check_diffs(used_bits, row, candidate, scope):
                            # Check subsequent elements
                            valid = True
                            for j in range(pos + 1, len(row)):
                                d = row[j] - candidate
                                if d <= 0 or used_bits[d]:
                                    valid = False
                                    break
                            if valid:
                                new_val = candidate
                                break

                    if new_val is not None:
                        row[pos] = new_val
                        mark_diffs(used_bits, row, new_val)
                        improved = True
                        any_progress = True
                    else:
                        mark_diffs(used_bits, row, old_val)

            if not any_progress:
                break

        if improved:
            cur_max = max(max(r) for r in rows)
            return rows, cur_max
        return None, None

    # --- Multi-start with diverse strategies ---
    def diverse_greedy(scope, rng, strategy="small"):
        """
        Greedy construction with a scope budget and a strategy for choosing candidates.
        """
        used_bits = make_bitset(scope)
        rows = []
        for row_idx in range(n_rows):
            if _time_left() < 3.0:
                return None
            row = [0]
            x = 0
            while len(row) < k + 1:
                valid = []
                for candidate in range(x + 1, scope + 1):
                    if check_diffs(used_bits, row, candidate, scope):
                        valid.append(candidate)
                        if len(valid) >= 20:
                            break

                if not valid:
                    return None

                if strategy == "small":
                    x = valid[0]
                elif strategy == "random":
                    x = valid[rng.randint(0, len(valid) - 1)]
                else:  # "weighted"
                    w = [1.0 / (i + 1) for i in range(len(valid))]
                    total_w = sum(w)
                    r = rng.random() * total_w
                    cum = 0.0
                    x = valid[-1]
                    for i in range(len(valid)):
                        cum += w[i]
                        if r <= cum:
                            x = valid[i]
                            break

                row.append(x)
                mark_diffs(used_bits, row, x)
            rows.append(row)

        return rows

    # --- Main search loop ---
    strategies = ["small", "random", "weighted"]

    # Phase 1: Try diverse greedy constructions with decreasing scope targets
    for scope_frac in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]:
        if _time_left() < 10.0:
            break

        target = int(best_max * scope_frac)
        if target < lower_bound:
            continue

        for strategy in strategies:
            for trial in range(3):
                if _time_left() < 5.0:
                    break
                rows = diverse_greedy(target, rng, strategy)
                if rows is None:
                    continue
                cur_max = max(max(r) for r in rows)
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = rows
                    # Apply local search
                    if _time_left() > 5.0:
                        improved_rows, improved_max = local_search_bitset(best_rows, best_max, max_iters=1000)
                        if improved_rows is not None and improved_max < best_max:
                            best_max = improved_max
                            best_rows = improved_rows
                    break

    # Phase 2: Simulated annealing from multiple starting points
    for attempt in range(20):
        if _time_left() < 10.0:
            break

        # Get a starting point via diverse greedy
        target = int(best_max * rng.uniform(0.8, 1.0))
        if target < lower_bound:
            target = lower_bound

        strategy = strategies[rng.randint(0, len(strategies) - 1)]
        rows = diverse_greedy(target, rng, strategy)
        if rows is None:
            # Fall back to best_rows
            rows = [list(r) for r in best_rows]

        sa_rows, sa_max = simulated_annealing(rows, best_max, max_iters=2000)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
            # Apply local search
            if _time_left() > 5.0:
                improved_rows, improved_max = local_search_bitset(best_rows, best_max, max_iters=1000)
                if improved_rows is not None and improved_max < best_max:
                    best_max = improved_max
                    best_rows = improved_rows

    # Phase 3: Intensive local search on best
    if _time_left() > 5.0:
        improved_rows, improved_max = local_search_bitset(best_rows, best_max, max_iters=3000)
        if improved_rows is not None and improved_max < best_max:
            best_max = improved_max
            best_rows = improved_rows

    if _time_left() > 3.0:
        improved_rows, improved_max = local_search_bitset(best_rows, best_max, max_iters=2000)
        if improved_rows is not None and improved_max < best_max:
            best_max = improved_max
            best_rows = improved_rows

    # Final validation
    def validate(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[j] <= row[i]:
                        return False
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    if not validate(best_rows):
        best_rows = greedy_construct()

    return best_rows
