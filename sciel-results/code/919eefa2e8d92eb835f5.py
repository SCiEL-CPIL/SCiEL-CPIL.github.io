"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math

_DEADLINE = None
_TIME_LIMIT = 115.0


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

    # --- Bitset-based conflict checking for speed ---
    # We represent the set of used differences as a boolean array
    # This allows O(1) per-difference checks

    def construct_with_bitset(target_scope, rng, bias_smallest=0.5):
        """
        Construct rows using bitset for fast conflict checking.
        Returns (rows, success).
        """
        # Maximum possible difference is target_scope
        # We need differences up to target_scope
        used_bits = bytearray(target_scope + 2)  # 1-indexed: used_bits[d] = 1 if d is used

        rows = []
        for row_idx in range(n_rows):
            if _time_left() < 3.0:
                return None, False
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Find valid candidates within [x+1, target_scope]
                valid = []
                for candidate in range(x + 1, target_scope + 1):
                    # Check if any difference candidate - a is already used
                    conflict = False
                    for a in row:
                        d = candidate - a
                        if d <= target_scope and used_bits[d]:
                            conflict = True
                            break
                    if not conflict:
                        valid.append(candidate)
                        if len(valid) >= 30:
                            break

                if not valid:
                    return None, False

                # Choose with bias
                if rng.random() < bias_smallest:
                    x = valid[0]
                else:
                    # Weighted toward smaller
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

                # Mark differences as used
                for a in row[:-1]:
                    d = x - a
                    if d <= target_scope:
                        used_bits[d] = 1

                row.append(x)
            rows.append(row)

        return rows, True

    # --- Local search with bitset ---
    def local_search_bitset(rows, target_scope, max_iters=200):
        """
        Local search to reduce max entry, using bitset for speed.
        Returns improved rows or None.
        """
        rows = [list(r) for r in rows]
        cur_max = max(max(r) for r in rows)

        # Build used set
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])

        improved = False
        for _ in range(max_iters):
            if _time_left() < 2.0:
                break

            # Process rows in order of largest entry
            row_order = sorted(range(len(rows)), key=lambda i: rows[i][-1], reverse=True)
            any_progress = False

            for max_idx in row_order:
                if _time_left() < 2.0:
                    break
                row = rows[max_idx]
                max_val = row[-1]

                # Try to reduce last element
                found = False
                for new_val in range(row[-2] + 1, max_val):
                    old_diffs = {row[-1] - a for a in row[:-1]}
                    new_diffs = {new_val - a for a in row[:-1]}
                    if new_diffs & (used - old_diffs):
                        continue
                    if new_val <= row[-2]:
                        continue
                    used -= old_diffs
                    used |= new_diffs
                    row[-1] = new_val
                    found = True
                    improved = True
                    any_progress = True
                    break

                if not found and len(row) >= 3:
                    # Try to reduce second-to-last and adjust last
                    pen_val = row[-2]
                    for new_pen in range(row[-3] + 1, pen_val):
                        old_diffs = set()
                        for a in row[:-2]:
                            old_diffs.add(pen_val - a)
                            old_diffs.add(row[-1] - a)
                        old_diffs.add(row[-1] - pen_val)

                        new_pen_diffs = {new_pen - a for a in row[:-2]}
                        if new_pen_diffs & (used - old_diffs):
                            continue
                        if new_pen <= row[-3]:
                            continue

                        best_last = None
                        for new_last in range(new_pen + 1, row[-1] + 1):
                            new_last_diffs = {new_last - a for a in row[:-2]}
                            new_last_diffs.add(new_last - new_pen)
                            all_new = new_pen_diffs | new_last_diffs
                            if all_new & (used - old_diffs):
                                continue
                            if best_last is None or new_last < best_last:
                                best_last = new_last

                        if best_last is not None:
                            used -= old_diffs
                            new_pen_diffs = {new_pen - a for a in row[:-2]}
                            new_last_diffs = {best_last - a for a in row[:-2]}
                            new_last_diffs.add(best_last - new_pen)
                            used |= new_pen_diffs
                            used |= new_last_diffs
                            row[-2] = new_pen
                            row[-1] = best_last
                            found = True
                            improved = True
                            any_progress = True
                            break

            if not any_progress:
                break

        if improved:
            return rows
        return None

    # --- Multi-start with varying biases and scopes ---
    def multi_start_search():
        nonlocal best_rows, best_max

        # Try different target scopes and biases
        base_scope = best_max
        scope_fracs = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]
        biases = [0.3, 0.5, 0.7, 0.9]

        for frac in scope_fracs:
            if _time_left() < 15.0:
                break
            target = int(base_scope * frac)
            if target < lower_bound:
                continue

            for bias in biases:
                if _time_left() < 10.0:
                    break

                for trial in range(3):
                    if _time_left() < 8.0:
                        break

                    rows, success = construct_with_bitset(target, rng, bias)
                    if not success:
                        continue

                    cur_max = max(max(r) for r in rows)
                    if cur_max < best_max:
                        best_max = cur_max
                        best_rows = rows

                        # Apply local search
                        if _time_left() > 5.0:
                            improved = local_search_bitset(best_rows, best_max, max_iters=150)
                            if improved:
                                new_max = max(max(r) for r in improved)
                                if new_max < best_max:
                                    best_max = new_max
                                    best_rows = improved

                        # Early exit if we hit the lower bound
                        if best_max <= lower_bound + 5:
                            return

    # --- Run multi-start search ---
    multi_start_search()

    # --- Final intensive local search ---
    if _time_left() > 5.0:
        improved = local_search_bitset(best_rows, best_max, max_iters=300)
        if improved:
            new_max = max(max(r) for r in improved)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved

    if _time_left() > 3.0:
        improved = local_search_bitset(best_rows, best_max, max_iters=200)
        if improved:
            new_max = max(max(r) for r in improved)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved

    # --- Validate ---
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
