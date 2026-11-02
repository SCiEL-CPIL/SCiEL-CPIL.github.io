"""Island-based hill-climbing with difference-set representation for difference-triangle sets.

Structural change: instead of a single random-walk over element positions, we maintain
a population of `n_islands` independent states. Each island performs local search with
two move types: (1) single-element shift, (2) full-row rebuild from scratch given the
global used-difference set. Islands periodically exchange their best solution with
neighboring islands (migration). This creates structured diversity and avoids the
single-trajectory trap.
"""

import time
import random
import sys


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 118.0
    m = k + 1  # entries per row
    n_diffs_per_row = m * (m - 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    def compute_used(rows):
        used = set()
        for row in rows:
            for i in range(m):
                for j in range(i + 1, m):
                    used.add(row[j] - row[i])
        return used

    def compute_scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(m):
                for j in range(i + 1, m):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_row(used, rng, max_val=200):
        """Build one row greedily, avoiding differences already in used."""
        row = [0]
        x = 0
        while len(row) < m and x < max_val:
            x += 1
            ok = True
            for a in row:
                if (x - a) in used:
                    ok = False
                    break
            if ok:
                row.append(x)
        if len(row) == m:
            return row
        return None

    def greedy_seed(rng, skip_prob=0.0):
        """Generate a full greedy seed."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            max_val = 300
            while len(row) < m and x < max_val:
                x += 1
                if skip_prob > 0 and rng.random() < skip_prob:
                    continue
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            if len(row) < m:
                return None
            rows.append(row)
        return rows

    def compact_row_right_to_left(row, used, rng):
        """Try to reduce each element from right to left to its minimum feasible value."""
        # We need to be careful: reducing an element changes differences with all others.
        # Strategy: for position p from m-1 down to 1, try to reduce row[p] to the
        # smallest value > row[p-1] that doesn't create conflicts.
        new_row = list(row)
        for p in range(m - 1, 0, -1):
            # Remove old diffs involving position p
            old_diffs = set()
            for i in range(p):
                old_diffs.add(new_row[p] - new_row[i])
            for j in range(p + 1, m):
                old_diffs.add(new_row[j] - new_row[p])
            for d in old_diffs:
                used.discard(d)

            # Try values from low to high
            low = new_row[p - 1] + 1
            found = False
            for candidate in range(low, new_row[p] + 1):
                # Check if candidate works
                new_diffs = set()
                ok = True
                for i in range(p):
                    d = candidate - new_row[i]
                    if d in used or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    for j in range(p + 1, m):
                        d = new_row[j] - candidate
                        if d in used or d in new_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                if ok:
                    new_row[p] = candidate
                    for d in new_diffs:
                        used.add(d)
                    found = True
                    break
            if not found:
                # Restore original
                new_row[p] = row[p]
                for d in old_diffs:
                    used.add(d)
            # If found, old_diffs are already discarded, new_diffs added
        return new_row

    def try_shift_element(rows, used, r, p, rng):
        """Try to shift element (r, p) to a nearby value. Returns (new_rows, new_used, improved)."""
        row = rows[r]
        old_val = row[p]
        low = row[p - 1] + 1
        high = row[p + 1] - 1 if p + 1 < m else old_val + 20

        # Remove old diffs
        old_diffs = []
        for i in range(p):
            old_diffs.append(row[p] - row[i])
        for j in range(p + 1, m):
            old_diffs.append(row[j] - row[p])
        for d in old_diffs:
            used.discard(d)

        # Try candidate values
        candidates = []
        # Try reducing
        for v in range(low, old_val):
            candidates.append(v)
        # Try small increases
        for v in range(old_val + 1, min(old_val + 15, high + 1)):
            candidates.append(v)
        rng.shuffle(candidates)

        for new_val in candidates:
            if new_val <= row[p - 1]:
                continue
            if p + 1 < m and new_val >= row[p + 1]:
                continue
            # Check conflicts
            new_diffs = []
            ok = True
            for i in range(p):
                d = new_val - row[i]
                if d in used:
                    ok = False
                    break
                new_diffs.append(d)
            if ok:
                for j in range(p + 1, m):
                    d = row[j] - new_val
                    if d in used:
                        ok = False
                    break
                new_diffs.append(d)
            if not ok:
                continue
            # Check internal distinctness
            if len(set(new_diffs)) != len(new_diffs):
                continue
            # Valid: apply
            new_row = list(row)
            new_row[p] = new_val
            new_rows = [list(r) for r in rows]
            new_rows[r] = new_row
            for d in new_diffs:
                used.add(d)
            return new_rows, used, True

        # Restore
        for d in old_diffs:
            used.add(d)
        return rows, used, False

    def rebuild_row(rows, used, r, rng):
        """Rebuild row r from scratch, keeping other rows fixed. Returns (new_rows, new_used, success)."""
        # Remove all diffs from row r
        row = rows[r]
        for i in range(m):
            for j in range(i + 1, m):
                used.discard(row[j] - row[i])

        # Try to build a new row
        new_row = greedy_row(used, rng, max_val=300)
        if new_row is None:
            # Restore old row
            for i in range(m):
                for j in range(i + 1, m):
                    used.add(row[j] - row[i])
            return rows, used, False

        new_rows = [list(r) for r in rows]
        new_rows[r] = new_row
        for i in range(m):
            for j in range(i + 1, m):
                used.add(new_row[j] - new_row[i])
        return new_rows, used, True

    # --- Initialize islands ---
    n_islands = 6
    rng = random.Random(12345)

    islands = []  # list of (rows, used, scope)
    best_rows = None
    best_scope = float('inf')

    for i in range(n_islands):
        seed = None
        for attempt in range(20):
            skip = rng.random() * 0.2
            seed = greedy_seed(rng, skip_prob=skip)
            if seed is not None:
                break
        if seed is None:
            # Fallback
            seed = greedy_seed(rng, skip_prob=0.0)
        if seed is None:
            # Shouldn't happen, but safety
            seed = greedy_seed(random.Random(0), skip_prob=0.0)
        used = compute_used(seed)
        scope = compute_scope(seed)
        islands.append([seed, used, scope])
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in seed]

    # --- Main loop: each island does local search, then migration ---
    step = 0
    migration_interval = 500

    while time.time() < deadline:
        for isl in range(n_islands):
            if time.time() > deadline:
                break

            rows, used, scope = islands[isl]

            # Local search: several random moves
            for _ in range(30):
                if time.time() > deadline:
                    break
                move_type = rng.random()
                if move_type < 0.6:
                    # Single element shift
                    r = rng.randrange(n_rows)
                    p = rng.randrange(1, m)
                    new_rows, new_used, improved = try_shift_element(rows, used, r, p, rng)
                    if improved:
                        new_scope = compute_scope(new_rows)
                        if new_scope <= scope:
                            rows, used, scope = new_rows, new_used, new_scope
                            if scope < best_scope:
                                best_scope = scope
                                best_rows = [list(r) for r in rows]
                elif move_type < 0.85:
                    # Row rebuild
                    r = rng.randrange(n_rows)
                    new_rows, new_used, success = rebuild_row(rows, used, r, rng)
                    if success:
                        new_scope = compute_scope(new_rows)
                        if new_scope <= scope:
                            rows, used, scope = new_rows, new_used, new_scope
                            if scope < best_scope:
                                best_scope = scope
                                best_rows = [list(r) for r in rows]
                else:
                    # Compact right-to-left on a random row
                    r = rng.randrange(n_rows)
                    new_row = compact_row_right_to_left(rows[r], used, rng)
                    if new_row != rows[r]:
                        new_rows = [list(r) for r in rows]
                        new_rows[r] = new_row
                        new_scope = compute_scope(new_rows)
                        if new_scope <= scope:
                            rows, used, scope = new_rows, used, new_scope
                            if scope < best_scope:
                                best_scope = scope
                                best_rows = [list(r) for r in rows]

            islands[isl] = [rows, used, scope]
            step += 1

        # Migration: exchange best with neighbor
        if step % migration_interval == 0:
            for isl in range(n_islands):
                neighbor = (isl + 1) % n_islands
                # Compare scopes
                if islands[isl][2] < islands[neighbor][2]:
                    # Copy isl's best to neighbor
                    islands[neighbor] = [[list(r) for r in islands[isl][0]],
                                          compute_used(islands[isl][0]),
                                          islands[isl][2]]
                else:
                    islands[isl] = [[list(r) for r in islands[neighbor][0]],
                                     compute_used(islands[neighbor][0]),
                                     islands[neighbor][2]]

    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
