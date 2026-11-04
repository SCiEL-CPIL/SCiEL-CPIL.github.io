"""Difference triangle set (7,5): minimize the largest entry.

Structural redesign:
- Bitmask difference tracking for O(1) conflict checks.
- Row-by-row backtracking construction (each row built with exact search).
- Multi-start with randomization over row order and element choices.
- Iterative greedy compression to reduce scope.
"""
import time
import random
import sys


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows.
    Minimize the largest entry."""
    deadline = time.time() + 140.0  # Safety margin before 150s kill

    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    count_bound = total_diffs  # m >= this

    # We need differences up to some max. Start with a generous upper bound.
    # For (7,5): total_diffs = 7*15 = 105, best known scope = 112.
    # Allow scope up to ~200 for safety.
    max_scope_initial = 250

    def build_bitmask(max_scope):
        """Return a bytearray of size max_scope+1, all zeros."""
        return bytearray(max_scope + 1)

    def row_diffs(row):
        """Compute all pairwise differences of a row."""
        diffs = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(row[j] - row[i])
        return diffs

    def valid_config(config):
        """Check if config is valid: all rows start at 0, strictly increasing,
        and all differences are distinct across all rows."""
        used = set()
        for row in config:
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            d = row_diffs(row)
            if len(set(d)) != len(d):
                return False
            if used & set(d):
                return False
            used |= set(d)
        return True

    def scope_of(config):
        return max(max(r) for r in config)

    def backtrack_row(used_mask, max_scope, row_order_seed):
        """Build a single row using backtracking.
        used_mask: bytearray where used_mask[d] = 1 if difference d is taken.
        Returns (row, success)."""
        row = [0]
        n_elems = k + 1

        def backtrack(pos, current_row, max_val):
            if pos == n_elems:
                return True

            # Determine range for next element
            low = current_row[-1] + 1
            high = max_val

            # Randomize order for diversity
            candidates = list(range(low, high + 1))
            random.shuffle(candidates)

            for val in candidates:
                # Check all differences with existing elements
                ok = True
                new_diffs = []
                for i in range(pos):
                    d = val - current_row[i]
                    if d > max_scope or used_mask[d]:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue

                # Check for duplicate differences within this row
                if len(set(new_diffs)) != len(new_diffs):
                    continue

                # Mark differences
                for d in new_diffs:
                    used_mask[d] = 1

                current_row.append(val)
                if backtrack(pos + 1, current_row, max_val):
                    return True
                current_row.pop()

                # Unmark
                for d in new_diffs:
                    used_mask[d] = 0

            return False

        if backtrack(1, row, max_scope):
            return row, True
        return None, False

    def greedy_row(used_mask, max_scope):
        """Build a row greedily: pick smallest valid value at each position."""
        row = [0]
        n_elems = k + 1

        for pos in range(1, n_elems):
            found = False
            for val in range(row[-1] + 1, max_scope + 1):
                ok = True
                new_diffs = []
                for i in range(pos):
                    d = val - row[i]
                    if d > max_scope or used_mask[d]:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue
                if len(set(new_diffs)) != len(new_diffs):
                    continue

                # Valid
                for d in new_diffs:
                    used_mask[d] = 1
                row.append(val)
                found = True
                break

            if not found:
                # Roll back: unmark differences from the last successful position
                # Actually we need to undo everything from this position onward
                # Since we're going forward, just return failure
                return None, False

        return row, True

    def random_row(used_mask, max_scope, spread):
        """Build a row with randomized choices, biased toward smaller values
        but with some spread for diversity."""
        row = [0]
        n_elems = k + 1

        for pos in range(1, n_elems):
            low = row[-1] + 1
            # Bias toward smaller values but allow some randomness
            # Use a geometric-like distribution
            base = low
            # Try a few random candidates, then fall back to greedy
            attempts = []
            for _ in range(5):
                # Random value, biased low
                r = random.expovariate(1.0 / max(1, spread))
                val = min(base + int(r), max_scope)
                if val < low:
                    val = low
                attempts.append(val)

            # Sort attempts to try smaller ones first, but with some randomness
            random.shuffle(attempts)
            # Also include greedy (smallest) as a candidate
            attempts.append(low)
            random.shuffle(attempts)

            found = False
            for val in attempts:
                if val < low or val > max_scope:
                    continue
                ok = True
                new_diffs = []
                for i in range(pos):
                    d = val - row[i]
                    if d > max_scope or used_mask[d]:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue
                if len(set(new_diffs)) != len(new_diffs):
                    continue

                for d in new_diffs:
                    used_mask[d] = 1
                row.append(val)
                found = True
                break

            if not found:
                # Fallback to greedy from current position
                for val in range(low, max_scope + 1):
                    ok = True
                    new_diffs = []
                    for i in range(pos):
                        d = val - row[i]
                        if d > max_scope or used_mask[d]:
                            ok = False
                            break
                        new_diffs.append(d)
                    if not ok:
                        continue
                    if len(set(new_diffs)) != len(new_diffs):
                        continue
                    for d in new_diffs:
                        used_mask[d] = 1
                    row.append(val)
                    found = True
                    break

            if not found:
                return None, False

        return row, True

    def compress_row(row, used_mask_excl_row, max_scope):
        """Try to compress a row by pulling elements down.
        used_mask_excl_row: bitmask of all differences EXCEPT this row's."""
        new_row = list(row)
        n = len(new_row)

        for pos in range(n - 1, 0, -1):
            old_val = new_row[pos]
            # Try smaller values
            for candidate in range(new_row[pos - 1] + 1, old_val):
                # Check conflicts
                ok = True
                new_diffs = []
                for i in range(pos):
                    d = candidate - new_row[i]
                    if d > max_scope or used_mask_excl_row[d]:
                        ok = False
                        break
                    new_diffs.append(d)
                # Also check against other elements in this row (pos+1 to end)
                for i in range(pos + 1, n):
                    d = new_row[i] - candidate
                    if d > max_scope or used_mask_excl_row[d]:
                        ok = False
                        break
                    new_diffs.append(d)

                if not ok:
                    continue
                if len(set(new_diffs)) != len(new_diffs):
                    continue

                new_row[pos] = candidate
                break

        return new_row

    def compress_all(config, max_scope):
        """Compress all rows iteratively."""
        for _ in range(5):
            changed = False
            for ridx in range(n_rows):
                # Compute used mask without this row
                used_mask = bytearray(max_scope + 1)
                for j, row in enumerate(config):
                    if j == ridx:
                        continue
                    for d in row_diffs(row):
                        if d <= max_scope:
                            used_mask[d] = 1

                new_row = compress_row(config[ridx], used_mask, max_scope)
                if new_row != config[ridx]:
                    config[ridx] = new_row
                    changed = True
            if not changed:
                break
        return config

    def try_construct(max_scope, mode, seed):
        """Try to construct a full config with given max_scope.
        mode: 'greedy', 'backtrack', 'random'
        Returns (config, success)."""
        random.seed(seed)
        used_mask = bytearray(max_scope + 1)
        config = []

        # Randomize row order for diversity
        row_order = list(range(n_rows))
        if mode != 'greedy':
            random.shuffle(row_order)

        for ridx in row_order:
            if mode == 'greedy':
                row, ok = greedy_row(used_mask, max_scope)
            elif mode == 'backtrack':
                row, ok = backtrack_row(used_mask, max_scope, seed + ridx)
            else:  # random
                row, ok = random_row(used_mask, max_scope, max_scope // 5)

            if not ok:
                return None, False
            config.append(row)

        # Reorder back to original
        if mode != 'greedy':
            final_config = [None] * n_rows
            for new_idx, orig_idx in enumerate(row_order):
                final_config[orig_idx] = config[new_idx]
            config = final_config

        return config, True

    def local_improve(config, max_scope, deadline):
        """Try to reduce scope via local search on individual elements."""
        current = [list(r) for r in config]
        best = [list(r) for r in current]
        best_scope = scope_of(current)

        # Build used mask
        used_mask = bytearray(max_scope + 1)
        for row in current:
            for d in row_diffs(row):
                if d <= max_scope:
                    used_mask[d] = 1

        T = float(best_scope)
        cooling = 0.9998

        while time.time() < deadline:
            # Pick random row and position
            ridx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            row = current[ridx]
            old_val = row[pos]

            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos + 1 < len(row) else best_scope + 3

            if low >= high:
                continue

            new_val = random.randint(low, high)
            if new_val == old_val:
                continue

            # Check validity of this move
            # Remove old diffs
            old_diffs = []
            for i in range(pos):
                old_diffs.append(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.append(row[i] - old_val)

            # Check new diffs
            new_diffs = []
            ok = True
            for i in range(pos):
                d = new_val - row[i]
                if d > max_scope:
                    ok = False
                    break
                new_diffs.append(d)
            if ok:
                for i in range(pos + 1, len(row)):
                    d = row[i] - new_val
                    if d > max_scope:
                        ok = False
                        break
                    new_diffs.append(d)

            if not ok or len(set(new_diffs)) != len(new_diffs):
                continue

            # Check conflicts with used_mask (excluding old diffs)
            conflict = False
            for d in new_diffs:
                if d in old_diffs:
                    continue
                if used_mask[d]:
                    conflict = True
                    break

            if conflict:
                # Accept with probability based on temperature
                n_conf = sum(1 for d in new_diffs if d not in old_diffs and used_mask[d])
                delta = n_conf * 10
                if random.random() < pow(2.718281828, -delta / max(T, 0.1)):
                    # Accept invalid move (will try to fix)
                    for d in old_diffs:
                        used_mask[d] = 0
                    row[pos] = new_val
                    for d in new_diffs:
                        used_mask[d] = 1
            else:
                # Valid move
                for d in old_diffs:
                    used_mask[d] = 0
                row[pos] = new_val
                for d in new_diffs:
                    used_mask[d] = 1

                new_scope = scope_of(current)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [list(r) for r in current]

            T *= cooling
            if T < 1.0:
                T = 1.0

        return best, best_scope

    # Main search loop
    best_config = None
    best_scope = float('inf')

    # Phase 1: Greedy baseline
    for max_scope in range(count_bound, max_scope_initial + 1):
        config, ok = try_construct(max_scope, 'greedy', 42)
        if ok:
            config = compress_all(config, max_scope)
            s = scope_of(config)
            if s < best_scope:
                best_scope = s
                best_config = [list(r) for r in config]
            break

    if best_config is None:
        # Fallback: greedy with larger scope
        config, ok = try_construct(max_scope_initial, 'greedy', 42)
        if ok:
            best_config = config
            best_scope = scope_of(config)

    # Phase 2: Multi-start with various modes and max_scope values
    if best_config is not None:
        # Compress the best
        best_config = compress_all(best_config, max(best_scope, count_bound))
        best_scope = scope_of(best_config)

    start_time = time.time()

    # Try different max_scope values and modes
    scopes_to_try = []
    if best_config is not None:
        # Try scopes around current best
        for s in range(max(count_bound, best_scope - 10), best_scope + 1):
            scopes_to_try.append(s)
    # Also try some fixed scopes
    for s in [count_bound, count_bound + 5, count_bound + 10, 110, 112, 115, 120]:
        if s >= count_bound and s not in scopes_to_try:
            scopes_to_try.append(s)

    scopes_to_try = sorted(set(scopes_to_try))

    modes = ['random', 'backtrack', 'greedy']
    seed_counter = 100

    for max_scope in scopes_to_try:
        if time.time() > deadline - 10:
            break
        if max_scope < count_bound:
            continue
        if max_scope < best_scope:
            # Only try if we can potentially improve
            pass
        else:
            # If max_scope >= best_scope, we can't improve scope, skip
            # unless we're trying to find a valid config at this scope
            if best_config is not None and max_scope >= best_scope:
                continue

        for mode in modes:
            if time.time() > deadline - 10:
                break
            # Try a few seeds
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                seed = seed_counter
                seed_counter += 1
                config, ok = try_construct(max_scope, mode, seed)
                if ok:
                    config = compress_all(config, max_scope)
                    s = scope_of(config)
                    if s < best_scope:
                        best_scope = s
                        best_config = [list(r) for r in config]
                        # Local improve
                        if time.time() < deadline - 30:
                            improved, imp_scope = local_improve(
                                best_config, max_scope, deadline - 30
                            )
                            if imp_scope < best_scope:
                                best_scope = imp_scope
                                best_config = improved

                # If we hit the counting bound, stop
                if best_scope <= count_bound:
                    break
            if best_scope <= count_bound:
                break
        if best_scope <= count_bound:
            break

    # Final compression
    if best_config is not None:
        max_sc = max(best_scope, count_bound)
        best_config = compress_all(best_config, max_sc)
        best_scope = scope_of(best_config)

    # Verify
    if best_config is not None and not valid_config(best_config):
        # Fallback
        config, ok = try_construct(max_scope_initial, 'greedy', 42)
        if ok:
            best_config = config
        else:
            # Last resort: construct with even larger scope
            for ms in range(max_scope_initial + 1, max_scope_initial + 100):
                config, ok = try_construct(ms, 'greedy', 42)
                if ok:
                    best_config = config
                    break

    if best_config is None:
        # Absolute fallback: simple greedy with large scope
        used_mask = bytearray(500)
        best_config = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(1, k + 1):
                for val in range(row[-1] + 1, 500):
                    ok = True
                    for i in range(pos):
                        d = val - row[i]
                        if d >= len(used_mask) or used_mask[d]:
                            ok = False
                            break
                    if ok:
                        for i in range(pos):
                            used_mask[val - row[i]] = 1
                        row.append(val)
                        break
            best_config.append(row)

    return best_config


# EVOLVE-BLOCK-END
