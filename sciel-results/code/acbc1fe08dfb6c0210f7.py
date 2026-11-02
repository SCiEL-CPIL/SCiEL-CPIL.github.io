import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 118.0
    m = k + 1  # entries per row
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        diffs = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_full(used, rng, max_val=None):
        """Build a full row greedily, avoiding used diffs. Returns row or None."""
        row = [0]
        for pos in range(k):
            found = False
            if max_val is not None:
                for _ in range(50):
                    x = rng.randint(row[-1] + 1, max_val)
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        found = True
                        break
            else:
                x = row[-1]
                while True:
                    x += 1
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        found = True
                        break
            if not found:
                return None
        return row

    def greedy_construct(n, k_val, rng):
        used = set()
        rows = []
        for _ in range(n):
            row = greedy_full(used, rng)
            if row is None:
                return None
            rows.append(row)
        return rows

    def greedy_variant(n, k_val, rng, bias):
        """Greedy with bias towards smaller or larger values."""
        used = set()
        rows = []
        for _ in range(n):
            row = [0]
            success = True
            for pos in range(k_val):
                found = False
                lo = row[-1] + 1
                # Try a range of candidates
                if bias < 0.5:
                    # Prefer smaller
                    for _ in range(30):
                        x = lo + int((rng.random() ** 2) * 50)
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            found = True
                            break
                else:
                    # Prefer larger (but bounded)
                    for _ in range(30):
                        x = lo + int(rng.random() * 80)
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            found = True
                            break
                if not found:
                    # Fallback to sequential
                    x = row[-1]
                    while True:
                        x += 1
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            break
            rows.append(row)
        return rows

    def compact_row(row, used, all_rows, my_index):
        """Try to reduce the largest entry of a row by shifting values down."""
        # Simple: try to reduce each element from the last one backward
        improved = False
        for pos in range(len(row) - 1, 0, -1):
            orig = row[pos]
            new_val = row[pos] - 1
            # Must be greater than row[pos-1]
            while new_val > row[pos - 1]:
                # Check if this new value works
                test_row = row[:pos] + [new_val] + row[pos+1:]
                # Check internal diffs are still distinct
                internal_diffs = set()
                valid_internal = True
                for i in range(len(test_row)):
                    for j in range(i + 1, len(test_row)):
                        d = test_row[j] - test_row[i]
                        if d in internal_diffs:
                            valid_internal = False
                            break
                        internal_diffs.add(d)
                    if not valid_internal:
                        break
                if not valid_internal:
                    new_val -= 1
                    continue
                # Check against other rows' diffs
                conflict = False
                for idx, other_row in enumerate(all_rows):
                    if idx == my_index:
                        continue
                    other_diffs = get_diffs(other_row)
                    for d in internal_diffs:
                        if d in other_diffs:
                            conflict = True
                            break
                    if conflict:
                        break
                if not conflict:
                    row[pos] = new_val
                    improved = True
                    break  # restart compaction from the top
                else:
                    new_val -= 1
        return improved

    def local_compact(rows, max_iter=100):
        """Iteratively compact rows to reduce their max entries."""
        for _ in range(max_iter):
            any_improved = False
            for i in range(len(rows)):
                if compact_row(rows[i], None, rows, i):
                    any_improved = True
            if not any_improved:
                break
        return rows

    def simulated_annealing(rows, rng, deadline, max_iter=5000):
        """SA with swap moves within rows to reduce scope."""
        if not rows:
            return rows
        rows = [list(r) for r in rows]
        current_scope = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]
        best_scope = current_scope
        temperature = 10.0
        cooling = 0.995

        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break

            # Pick a random row and a random position to try changing
            row_idx = rng.randint(0, len(rows) - 1)
            row = rows[row_idx]
            pos = rng.randint(1, len(row) - 1)  # don't touch position 0

            # Try a small perturbation
            delta = rng.randint(-3, 3)
            new_val = row[pos] + delta

            if new_val <= row[pos - 1] or (pos < len(row) - 1 and new_val >= row[pos + 1]):
                continue

            # Check if this change is valid for the row
            test_row = row[:]
            test_row[pos] = new_val
            internal_diffs = set()
            valid = True
            for i in range(len(test_row)):
                for j in range(i + 1, len(test_row)):
                    d = test_row[j] - test_row[i]
                    if d in internal_diffs:
                        valid = False
                        break
                    internal_diffs.add(d)
                if not valid:
                    break

            if not valid:
                continue

            # Check against other rows
            conflict = False
            for idx, other_row in enumerate(rows):
                if idx == row_idx:
                    continue
                other_diffs = get_diffs(other_row)
                for d in internal_diffs:
                    if d in other_diffs:
                        conflict = True
                        break
                if conflict:
                    break

            if conflict:
                continue

            # Accept or reject based on SA criterion
            new_scope = max(max(r) for r in rows if r is not test_row)
            new_scope = max(new_scope, test_row[-1])
            if new_scope < current_scope:
                # Always accept improvement
                rows[row_idx] = test_row
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
            else:
                # Accept with probability exp(-(new_scope - current_scope)/temperature)
                if rng.random() < math.exp(-1.0 / max(temperature, 0.01)):
                    rows[row_idx] = test_row
                    current_scope = new_scope

            temperature *= cooling

        return best_rows

    # Main
    rng = random.Random(12345)

    # Initial greedy solution
    best_rows = greedy_construct(n_rows, k, rng)
    if best_rows is None:
        # Fallback: very simple sequential
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            best_rows.append(row)

    best_scope = max(max(r) for r in best_rows)

    # Try multiple greedy variants
    for trial in range(10):
        if time.time() > deadline - 30:
            break
        bias = rng.random()
        rows = greedy_variant(n_rows, k, rng, bias)
        if rows and is_valid(rows):
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_rows = rows
                best_scope = scope

    # Local compaction
    best_rows = local_compact(best_rows, max_iter=50)
    if is_valid(best_rows):
        best_scope = max(max(r) for r in best_rows)
    else:
        # Revert if invalid
        best_rows = greedy_construct(n_rows, k, rng)
        best_scope = max(max(r) for r in best_rows)

    # Simulated annealing
    best_rows = simulated_annealing(best_rows, rng, deadline, max_iter=10000)
    if is_valid(best_rows):
        best_scope = max(max(r) for r in best_rows)
    else:
        # Revert
        best_rows = greedy_construct(n_rows, k, rng)
        best_scope = max(max(r) for r in best_rows)

    # More diverse restarts with SA
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        bias = rng.random()
        rows = greedy_variant(n_rows, k, rng, bias)
        if rows and is_valid(rows):
            rows = simulated_annealing(rows, rng, deadline, max_iter=5000)
            if is_valid(rows):
                scope = max(max(r) for r in rows)
                if scope < best_scope:
                    best_rows = rows
                    best_scope = scope

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
