import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0

    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs  # Lower bound on scope (max entry must be >= total_diffs)

    best_rows = None
    best_scope = float('inf')

    def validate_rows(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def compute_used_set(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_construct(seed, target_scope=None):
        """
        Construct rows using greedy with randomization.
        If target_scope is given, try to keep all elements <= target_scope.
        """
        rng = random.Random(seed)
        used = set()
        rows = []

        max_scope = target_scope if target_scope else lb + 200

        for row_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False

                # Try compact first, then spread
                search_end = min(last + 30 + rng.randint(0, 20), max_scope)

                candidates = list(range(last + 1, search_end + 1))
                rng.shuffle(candidates)

                for x in candidates:
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break

                if not found:
                    # Try wider search
                    for x in range(last + 1, max_scope + 1):
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

    def random_restart_construct(seed, target_scope=None):
        """Build with random jumps."""
        rng = random.Random(seed)
        used = set()
        rows = []
        max_scope = target_scope if target_scope else lb + 300

        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                # Random jump size
                jump = rng.randint(2, 50)
                x = last + jump

                for candidate in range(max(last + 1, x - 15), min(max_scope + 1, x + 15)):
                    new_diffs = [candidate - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(candidate)
                    used.update(new_diffs)
                    found = True
                    break

                if not found:
                    for candidate in range(last + 1, max_scope + 1):
                        new_diffs = [candidate - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(candidate)
                        used.update(new_diffs)
                        found = True
                        break

                if not found:
                    return None
            rows.append(row)
        return rows

    def local_improve(rows, seed, max_iters=500):
        """Try to reduce scope by shifting elements left."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        used_set = compute_used_set(rows)

        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break

            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue

            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue

            # Remove this element and its diffs from used_set
            for a in row[:elem_idx]:
                used_set.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used_set.discard(a - old_val)
            row.pop(elem_idx)

            # Try to re-insert at a smaller position
            found = False
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1

            for new_val in range(max_val, min_val - 1, -1):
                valid = True
                for a in row[:elem_idx]:
                    d = new_val - a
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    for a in row[elem_idx:]:
                        d = a - new_val
                        if d in used_set:
                            valid = False
                            break
                if valid:
                    row.insert(elem_idx, new_val)
                    for a in row[:elem_idx]:
                        used_set.add(new_val - a)
                    for a in row[elem_idx + 1:]:
                        used_set.add(a - new_val)
                    found = True
                    break

            if found:
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used_set.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used_set.add(a - old_val)

        return rows, scope

    def mrv_backtrack_polish(rows, seed, max_time=5.0):
        """
        MRV-style backtracking to pack elements as tightly as possible.
        """
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        start = time.time()

        used_set = compute_used_set(rows)
        scope = get_scope(rows)

        for row_idx in range(n_rows):
            if time.time() - start > max_time:
                break

            row = rows[row_idx]
            for elem_idx in range(len(row) - 1, 0, -1):
                if time.time() - start > max_time:
                    break

                old_val = row[elem_idx]
                if old_val <= 1:
                    continue

                for a in row[:elem_idx]:
                    used_set.discard(old_val - a)
                for a in row[elem_idx + 1:]:
                    used_set.discard(a - old_val)
                row.pop(elem_idx)

                min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
                max_val = old_val

                found = False
                for new_val in range(min_val, max_val + 1):
                    valid = True
                    for a in row[:elem_idx]:
                        d = new_val - a
                        if d in used_set:
                            valid = False
                            break
                    if valid:
                        for a in row[elem_idx:]:
                            d = a - new_val
                            if d in used_set:
                                valid = False
                                break
                    if valid:
                        row.insert(elem_idx, new_val)
                        for a in row[:elem_idx]:
                            used_set.add(new_val - a)
                        for a in row[elem_idx + 1:]:
                            used_set.add(a - new_val)
                        found = True
                        break

                if not found:
                    row.insert(elem_idx, old_val)
                    for a in row[:elem_idx]:
                        used_set.add(old_val - a)
                    for a in row[elem_idx + 1:]:
                        used_set.add(a - old_val)

        new_scope = get_scope(rows)
        return rows, new_scope

    def simulated_annealing_improve(rows, seed, max_iters=300, t_start=5.0, t_end=0.1):
        """Simulated annealing to escape local minima."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        current_scope = get_scope(rows)
        best_scope_sa = current_scope
        best_rows_sa = [list(r) for r in rows]

        used_set = compute_used_set(rows)

        for it in range(max_iters):
            if time.time() > deadline - 2:
                break

            t = t_start * (t_end / t_start) ** (it / max_iters)

            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue

            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue

            for a in row[:elem_idx]:
                used_set.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used_set.discard(a - old_val)
            row.pop(elem_idx)

            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = row[elem_idx] - 1 if elem_idx < len(row) else current_scope + 5

            if min_val >= max_val:
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used_set.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used_set.add(a - old_val)
                continue

            found = False
            new_val = None
            for _ in range(20):
                v = rng.randint(min_val, max_val)
                valid = True
                for a in row[:elem_idx]:
                    d = v - a
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    for a in row[elem_idx:]:
                        d = a - v
                        if d in used_set:
                            valid = False
                            break
                if valid:
                    new_val = v
                    found = True
                    break

            if found:
                row.insert(elem_idx, new_val)
                for a in row[:elem_idx]:
                    used_set.add(new_val - a)
                for a in row[elem_idx + 1:]:
                    used_set.add(a - new_val)

                new_scope = get_scope(rows)
                delta = new_scope - current_scope

                if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(t, 1e-6)):
                    current_scope = new_scope
                    if new_scope < best_scope_sa:
                        best_scope_sa = new_scope
                        best_rows_sa = [list(r) for r in rows]
                else:
                    row.pop(elem_idx)
                    for a in row[:elem_idx]:
                        used_set.discard(new_val - a)
                    for a in row[elem_idx + 1:]:
                        used_set.discard(a - new_val)
                    row.insert(elem_idx, old_val)
                    for a in row[:elem_idx]:
                        used_set.add(old_val - a)
                    for a in row[elem_idx + 1:]:
                        used_set.add(a - old_val)
            else:
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used_set.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used_set.add(a - old_val)

        return best_rows_sa, best_scope_sa

    # Strategy 1: Many greedy attempts with target scopes
    for target in range(lb, lb + 50):
        if time.time() > deadline - 10:
            break
        for seed in range(200):
            if time.time() > deadline - 5:
                break
            rows = greedy_construct(seed * 31 + 7, target)
            if rows is not None:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                    if best_scope == lb:
                        break
        if best_scope == lb:
            break

    # Strategy 2: Random restarts
    for i in range(500):
        if time.time() > deadline - 5:
            break
        target = lb + (i % 30)
        rows = random_restart_construct(100000 + i * 97, target)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # Strategy 3: Local improvement
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 42, max_iters=1000)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Strategy 4: Simulated annealing
    if best_rows is not None and time.time() < deadline - 10:
        sa_rows, sa_scope = simulated_annealing_improve(best_rows, 12345, max_iters=300, t_start=5.0, t_end=0.1)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows

    # Strategy 5: More local improvement
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 12345, max_iters=500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Strategy 6: More simulated annealing
    if best_rows is not None and time.time() < deadline - 5:
        sa_rows, sa_scope = simulated_annealing_improve(best_rows, 99999, max_iters=200, t_start=3.0, t_end=0.05)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows

    # Strategy 7: MRV backtracking polish
    if best_rows is not None and time.time() < deadline - 3:
        polished, pol_scope = mrv_backtrack_polish(best_rows, 55555, max_time=3.0)
        if pol_scope < best_scope:
            best_scope = pol_scope
            best_rows = polished

    # Final Compaction
    if best_rows is not None and time.time() < deadline - 2:
        improved, imp_scope = local_improve(best_rows, 77777, max_iters=300)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    return best_rows


if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
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
    else:
        print("No valid construction found")
