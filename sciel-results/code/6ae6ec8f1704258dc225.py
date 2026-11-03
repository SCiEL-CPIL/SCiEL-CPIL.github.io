import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0

    def _scope(rows):
        return max(a for r in rows for a in r[1:])

    def _diffs_of_row(row):
        """Return list of all pairwise differences for a row."""
        diffs = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(row[j] - row[i])
        return diffs

    def _greedy_construct(n_rows, k, seed=0):
        rng = random.Random(seed)
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

    def _greedy_construct_randomized(n_rows, k, seed=0):
        """Greedy with randomized choice among valid next values."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try next few values, pick a random valid one
                candidates = []
                for _trial in range(3):
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                if candidates:
                    chosen = rng.choice(candidates)
                    # Need to skip to chosen
                    # Actually simpler: just try sequentially but with random offset
                    break
                # If no candidates in this batch, keep going
                if not candidates:
                    continue
            # Fallback: standard greedy if above logic fails
            if len(row) < k + 1:
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

    def _greedy_construct_skip(n_rows, k, seed=0, skip_range=(0, 3)):
        """Greedy that randomly skips some candidate values."""
        rng = random.Random(seed)
        lo, hi = skip_range
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Randomly skip some values
                if rng.random() < 0.15:
                    continue
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def _incremental_local_search(rows, n_rows, k, deadline, seed=42):
        """Fast local search with incremental diff-count updates."""
        best_rows = [list(r) for r in rows]
        best_scope = _scope(best_rows)
        current_rows = [list(r) for r in rows]
        current_scope = _scope(current_rows)

        # Build global diff counts
        diff_counts = Counter()
        for r in current_rows:
            for d in _diffs_of_row(r):
                diff_counts[d] += 1

        # Count collisions
        def _collision_count(counts):
            return sum(c - 1 for c in counts.values() if c > 1)

        current_collisions = _collision_count(diff_counts)
        current_cost = current_scope + 1000 * current_collisions
        best_cost = current_cost

        rng = random.Random(seed)
        temp = 50.0
        cooling = 0.9999
        min_temp = 0.5
        steps_since_improve = 0
        max_stall = 5000

        while time.time() < deadline - 3 and temp > min_temp:
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            delta = rng.randint(-10, 10)
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val <= 0:
                continue
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue

            # Compute old diffs for this row
            old_diffs = _diffs_of_row(row)
            # Modify row
            row[pos] = new_val
            new_diffs = _diffs_of_row(row)
            new_scope_val = max(row) if max(row) > current_scope else current_scope
            # Recompute scope properly
            new_scope = max(max(r) for r in current_rows)

            # Update diff counts incrementally
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1

            new_collisions = _collision_count(diff_counts)
            new_cost = new_scope + 1000 * new_collisions
            delta_c = new_cost - current_cost

            if delta_c < 0 or rng.random() < math.exp(-delta_c / temp):
                current_cost = new_cost
                current_scope = new_scope
                current_collisions = new_collisions
                steps_since_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_rows = [list(r) for r in current_rows]
            else:
                # Revert
                row[pos] = old_val
                # Revert diff counts
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1

            steps_since_improve += 1
            if steps_since_improve > max_stall:
                # Reset to best and restart
                current_rows = [list(r) for r in best_rows]
                current_scope = _scope(current_rows)
                current_collisions = _collision_count(diff_counts)
                current_cost = current_scope + 1000 * current_collisions
                temp = 50.0
                steps_since_improve = 0
                max_stall = 3000

            temp *= cooling

        return best_rows

    def _multi_row_local_search(rows, n_rows, k, deadline, seed=0):
        """Local search that also tries swapping elements between rows."""
        best_rows = [list(r) for r in rows]
        best_scope = _scope(best_rows)
        current_rows = [list(r) for r in rows]

        diff_counts = Counter()
        for r in current_rows:
            for d in _diffs_of_row(r):
                diff_counts[d] += 1

        def _collision_count(counts):
            return sum(c - 1 for c in counts.values() if c > 1)

        current_collisions = _collision_count(diff_counts)
        current_scope = _scope(current_rows)
        current_cost = current_scope + 1000 * current_collisions
        best_cost = current_cost

        rng = random.Random(seed)
        temp = 30.0
        cooling = 0.99995
        min_temp = 0.5
        stall = 0
        max_stall = 8000

        while time.time() < deadline - 3 and temp > min_temp:
            move_type = rng.random()

            if move_type < 0.6:
                # Single element shift within a row
                ri = rng.randrange(n_rows)
                row = current_rows[ri]
                pos = rng.randrange(1, len(row))
                old_val = row[pos]
                delta = rng.randint(-12, 12)
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue

                old_diffs = _diffs_of_row(row)
                row[pos] = new_val
                new_diffs = _diffs_of_row(row)
                new_scope = max(max(r) for r in current_rows)

                for d in old_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                for d in new_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1

                new_collisions = _collision_count(diff_counts)
                new_cost = new_scope + 1000 * new_collisions
                delta_c = new_cost - current_cost

                if delta_c < 0 or rng.random() < math.exp(-delta_c / temp):
                    current_cost = new_cost
                    current_scope = new_scope
                    current_collisions = new_collisions
                    stall = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_rows = [list(r) for r in current_rows]
                else:
                    row[pos] = old_val
                    for d in new_diffs:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                    for d in old_diffs:
                        diff_counts[d] = diff_counts.get(d, 0) + 1

            elif move_type < 0.8 and n_rows >= 2:
                # Swap elements between two rows
                ri, rj = rng.sample(range(n_rows), 2)
                row_i = current_rows[ri]
                row_j = current_rows[rj]
                pi = rng.randrange(1, len(row_i))
                pj = rng.randrange(1, len(row_j))
                vi = row_i[pi]
                vj = row_j[pj]

                # Check ordering constraints
                if pi > 0 and vj <= row_i[pi - 1]:
                    continue
                if pi < len(row_i) - 1 and vj >= row_i[pi + 1]:
                    continue
                if pj > 0 and vi <= row_j[pj - 1]:
                    continue
                if pj < len(row_j) - 1 and vi >= row_j[pj + 1]:
                    continue

                old_diffs_i = _diffs_of_row(row_i)
                old_diffs_j = _diffs_of_row(row_j)

                row_i[pi] = vj
                row_j[pj] = vi

                new_diffs_i = _diffs_of_row(row_i)
                new_diffs_j = _diffs_of_row(row_j)
                new_scope = max(max(r) for r in current_rows)

                for d in old_diffs_i:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                for d in old_diffs_j:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                for d in new_diffs_i:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                for d in new_diffs_j:
                    diff_counts[d] = diff_counts.get(d, 0) + 1

                new_collisions = _collision_count(diff_counts)
                new_cost = new_scope + 1000 * new_collisions
                delta_c = new_cost - current_cost

                if delta_c < 0 or rng.random() < math.exp(-delta_c / temp):
                    current_cost = new_cost
                    current_scope = new_scope
                    current_collisions = new_collisions
                    stall = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_rows = [list(r) for r in current_rows]
                else:
                    row_i[pi] = vi
                    row_j[pj] = vj
                    for d in new_diffs_i:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                    for d in new_diffs_j:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                    for d in old_diffs_i:
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    for d in old_diffs_j:
                        diff_counts[d] = diff_counts.get(d, 0) + 1

            elif move_type < 0.9 and n_rows >= 2:
                # Rebuild one row greedily given current used diffs
                ri = rng.randrange(n_rows)
                old_row = current_rows[ri]
                old_diffs = _diffs_of_row(old_row)

                # Remove old row's diffs
                for d in old_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]

                # Greedily build new row
                used = set()
                for d, c in diff_counts.items():
                    if c > 0:
                        used.add(d)

                new_row = [0]
                x = 0
                while len(new_row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in new_row}
                    if not new_diffs & used:
                        new_row.append(x)
                        used |= new_diffs

                current_rows[ri] = new_row
                new_diffs = _diffs_of_row(new_row)
                new_scope = max(max(r) for r in current_rows)

                for d in new_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1

                new_collisions = _collision_count(diff_counts)
                new_cost = new_scope + 1000 * new_collisions
                delta_c = new_cost - current_cost

                if delta_c < 0 or rng.random() < math.exp(-delta_c / temp):
                    current_cost = new_cost
                    current_scope = new_scope
                    current_collisions = new_collisions
                    stall = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_rows = [list(r) for r in current_rows]
                else:
                    # Revert
                    for d in new_diffs:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                    current_rows[ri] = old_row
                    for d in old_diffs:
                        diff_counts[d] = diff_counts.get(d, 0) + 1

            stall += 1
            if stall > max_stall:
                current_rows = [list(r) for r in best_rows]
                current_scope = _scope(current_rows)
                current_collisions = _collision_count(diff_counts)
                current_cost = current_scope + 1000 * current_collisions
                temp = 30.0
                stall = 0
                max_stall = 5000

            temp *= cooling

        return best_rows

    # Phase 1: Multi-start greedy with different strategies
    best_rows = None
    best_scope = float('inf')

    # Standard greedy
    for seed in range(40):
        if time.time() >= deadline - 60:
            break
        rows = _greedy_construct(n_rows, k, seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]

    # Skip-based greedy
    for seed in range(30):
        if time.time() >= deadline - 60:
            break
        rows = _greedy_construct_skip(n_rows, k, seed, skip_range=(0, 5))
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]

    if best_rows is None:
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

    # Phase 2: Incremental local search (fast, many moves)
    for ls_seed in range(4):
        if time.time() >= deadline - 20:
            break
        ls_rows = _incremental_local_search(best_rows, n_rows, k, deadline - 15, seed=ls_seed * 7919 + 13)
        ls_scope = _scope(ls_rows)
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = ls_rows

    # Phase 3: Multi-row local search with swaps
    if time.time() < deadline - 5:
        for mr_seed in range(3):
            if time.time() >= deadline - 5:
                break
            mr_rows = _multi_row_local_search(best_rows, n_rows, k, deadline - 3, seed=mr_seed * 104729 + 7)
            mr_scope = _scope(mr_rows)
            if mr_scope < best_scope:
                best_scope = mr_scope
                best_rows = mr_rows

    # Phase 4: Final polish - try to reduce scope by shifting
    if time.time() < deadline - 2:
        rng = random.Random(12345)
        for _ in range(50000):
            if time.time() >= deadline - 2:
                break
            ri = rng.randrange(n_rows)
            row = best_rows[ri]
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                row[pos] = new_val
                new_scope = _scope(best_rows)
                # Check validity
                seen = set()
                valid = True
                for r in best_rows:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            d = r[j] - r[i]
                            if d <= 0 or d in seen:
                                valid = False
                                break
                            seen.add(d)
                        if not valid:
                            break
                    if not valid:
                        break
                if valid and new_scope < best_scope:
                    best_scope = new_scope
                    break
                else:
                    row[pos] = old_val

    return best_rows
