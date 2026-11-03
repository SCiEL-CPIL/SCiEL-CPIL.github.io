import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound: need total_diffs distinct positive integers, max >= total_diffs
    lb = total_diffs

    def time_left():
        return deadline - time.time()

    def backtrack_scope(M, seed, rng=None):
        """Try to find n_rows valid rows within scope [0, M] using randomized backtracking."""
        if rng is None:
            rng = random.Random(seed)

        # Precompute difference sets for all valid rows within [0, M]
        # A row is a sorted list of length k+1 starting at 0, ending <= M
        # Its difference set must be unique (no internal collisions)
        all_rows = []
        all_diff_sets = []  # frozenset of differences for each row

        # We'll use a different approach: build rows one at a time with backtracking
        # but track which differences are used globally
        used = [False] * (M + 1)  # used[d] = True if difference d is taken
        used_count = 0
        rows = []

        def gen_rows_with_backtracking(row_idx, max_steps=50000):
            """Recursively build row row_idx, trying random valid candidates."""
            if row_idx == n_rows:
                return True

            if time_left() < 1.0:
                return False

            # Generate candidate rows: random increasing sequences of length k+1 in [0, M]
            # that start at 0 and have no internal difference collisions
            attempts = 0
            max_attempts = 20000

            while attempts < max_attempts and time_left() > 1.0:
                attempts += 1

                # Generate random row
                # Choose k-1 intermediate values from [1, M-1] (excluding 0 and M)
                if k - 1 == 0:
                    # k=1: row is just [0, x] for some x in [1, M]
                    candidates = list(range(1, M + 1))
                    rng.shuffle(candidates)
                    for x in candidates:
                        d = x
                        if not used[d]:
                            row = [0, x]
                            rows.append(row)
                            used[d] = True
                            if gen_rows_with_backtracking(row_idx + 1):
                                return True
                            used[d] = False
                            rows.pop()
                    return False
                else:
                    # Choose k-1 values from [1, M]
                    # Use random sampling for diversity
                    if M - 1 >= k - 1:
                        intermediates = rng.sample(range(1, M + 1), k - 1)
                        intermediates.sort()
                        row = [0] + intermediates
                    else:
                        # M too small for this k
                        return False

                    # Check internal difference uniqueness
                    diffs = set()
                    valid = True
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in diffs:
                                valid = False
                                break
                            diffs.add(d)
                    if not valid:
                        continue

                    # Check global difference uniqueness
                    all_used = True
                    for d in diffs:
                        if used[d]:
                            all_used = False
                            break
                    if not all_used:
                        continue

                    # Commit
                    rows.append(row)
                    for d in diffs:
                        used[d] = True

                    if gen_rows_with_backtracking(row_idx + 1):
                        return True

                    # Undo
                    rows.pop()
                    for d in diffs:
                        used[d] = False

            return False

        if gen_rows_with_backtracking(0):
            return rows
        return None

    def compaction_pass(rows, max_rounds=100):
        """Aggressively try to reduce each element to the smallest valid position."""
        rows = [list(r) for r in rows]
        n = n_rows

        for rnd in range(max_rounds):
            if time_left() < 3.0:
                break

            improved = False

            for row_idx in range(n):
                row = rows[row_idx]
                for pos in range(len(row) - 1, 0, -1):
                    old_val = row[pos]
                    if old_val <= 1:
                        continue

                    # Remove element at pos
                    row.pop(pos)

                    # Build used set from all rows (without this element)
                    used = set()
                    for r in rows:
                        for i in range(len(r)):
                            for j in range(i + 1, len(r)):
                                used.add(r[j] - r[i])

                    # Try to find smallest valid position
                    min_val = row[pos - 1] + 1 if pos > 0 else 1
                    found = False
                    for new_val in range(min_val, old_val):
                        valid = True
                        # Check diffs with elements before pos
                        for a in row[:pos]:
                            d = new_val - a
                            if d in used:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Check diffs with elements from pos onwards
                        for a in row[pos:]:
                            d = a - new_val
                            if d in used:
                                valid = False
                                break
                        if valid:
                            row.insert(pos, new_val)
                            found = True
                            improved = True
                            break

                    if not found:
                        row.insert(pos, old_val)

            if not improved:
                break

        return rows

    def greedy_build(seed):
        """Fast greedy: build rows sequentially, choosing smallest valid next element."""
        rng = random.Random(seed)
        used = set()
        rows = []

        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                # Try from last+1 upwards
                for x in range(last + 1, last + 200):
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    return None
            rows.append(row)
        return rows

    def greedy_build_random(seed, spread_factor=1.0):
        """Greedy with randomization: sometimes skip ahead."""
        rng = random.Random(seed)
        used = set()
        rows = []

        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                # Randomized search window
                window = int(50 * spread_factor) + rng.randint(0, 30)
                for x in range(last + 1, last + window):
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    for x in range(last + 1, last + 300):
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                    if not found:
                        return None
            rows.append(row)
        return rows

    best_rows = None
    best_scope = float('inf')

    # Phase 1: Quick greedy constructions to get a starting point
    for i in range(50):
        if time_left() < 10.0:
            break
        rows = greedy_build(i * 13 + 5)
        if rows is not None:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]

    # Phase 2: Randomized greedy with different parameters
    for i in range(100):
        if time_left() < 15.0:
            break
        spread = 0.7 + (i % 10) * 0.1
        rows = greedy_build_random(1000 + i * 37, spread)
        if rows is not None:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]

    # Phase 3: Compaction on best found
    if best_rows is not None and time_left() > 10.0:
        compacted = compaction_pass(best_rows, max_rounds=200)
        scope = max(max(r) for r in compacted)
        if scope < best_scope:
            best_scope = scope
            best_rows = compacted

    # Phase 4: Iterative deepening backtracking - try to find solution with smaller scope
    # Start from current best and try to reduce
    if time_left() > 20.0:
        # Try backtracking with decreasing scope targets
        target_scope = best_scope - 1
        max_target = lb

        while target_scope >= max(max_target, lb) and time_left() > 10.0:
            # Try multiple random seeds for this scope
            found = False
            for seed in range(50):
                if time_left() < 5.0:
                    break
                rng = random.Random(100000 + target_scope * 1000 + seed)
                rows = backtrack_scope(target_scope, seed, rng)
                if rows is not None:
                    scope = max(max(r) for r in rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in rows]
                    found = True
                    break
            if found:
                # Compact the new best
                if time_left() > 5.0:
                    compacted = compaction_pass(best_rows, max_rounds=100)
                    scope = max(max(r) for r in compacted)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = compacted
                target_scope = best_scope - 1
            else:
                # Couldn't find at this scope, try next
                target_scope -= 1

    # Phase 5: Final compaction
    if best_rows is not None and time_left() > 5.0:
        compacted = compaction_pass(best_rows, max_rounds=300)
        scope = max(max(r) for r in compacted)
        if scope < best_scope:
            best_scope = scope
            best_rows = compacted

    return best_rows


if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        scope = max(max(r) for r in result)
        print(f"Scope: {scope}")
        print(f"Rows: {result}")
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
        print(f"Total diffs used: {len(used)}")
    else:
        print("No valid construction found")
