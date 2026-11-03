import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on scope
    lb = total_diffs + 1

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def greedy_construct(seed_val, max_scope=None):
        """Construct rows using greedy with a compact strategy."""
        if max_scope is None:
            max_scope = lb + 100
        random.seed(seed_val)
        used = set()
        rows = []

        for row_idx in range(n_rows):
            row = [0]

            while len(row) < k + 1:
                last = row[-1]
                found = False
                window = 20 + len(row) * 8
                search_end = min(last + window, max_scope)

                candidates = list(range(last + 1, search_end + 1))
                random.shuffle(candidates)

                for x in candidates:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)

                    if valid:
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break

                if not found:
                    # Expand search
                    for x in range(last + 1, max_scope + 1):
                        valid = True
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d in used:
                                valid = False
                                break
                            new_diffs.append(d)

                        if valid:
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                    if not found:
                        return None, float('inf')

            rows.append(row)

        scope = max(max(r) for r in rows)
        return rows, scope

    # 1. Get initial solution
    best_rows = None
    best_scope = float('inf')

    for s in range(50):
        if time.time() > deadline - 60:
            break
        rows, scope = greedy_construct(s)
        if rows and scope < best_scope:
            best_scope = scope
            best_rows = rows

    if best_rows is None:
        # Fallback: Simple greedy with large search space
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            best_rows.append(row)
        best_scope = max(max(r) for r in best_rows)

    # 2. Constraint-based local search with forward checking
    # Represent each row as list of k gaps
    # row[i] = sum of gaps[0..i-1]
    
    def rows_to_gaps(rows):
        gaps = []
        for r in rows:
            g = [r[i+1] - r[i] for i in range(len(r)-1)]
            gaps.append(g)
        return gaps

    def gaps_to_rows(gaps):
        rows = []
        for g in gaps:
            row = [0]
            for gap in g:
                row.append(row[-1] + gap)
            rows.append(row)
        return rows

    def compute_all_diffs(gaps):
        """Compute all pairwise differences for each row from gaps."""
        all_diffs = []
        for g in gaps:
            row = [0]
            for gap in g:
                row.append(row[-1] + gap)
            diffs = []
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.append(row[j] - row[i])
            all_diffs.append(diffs)
        return all_diffs

    def build_diff_counts(all_diffs):
        """Build a dictionary mapping diff -> count."""
        counts = {}
        for diffs in all_diffs:
            for d in diffs:
                if d in counts:
                    counts[d] += 1
                else:
                    counts[d] = 1
        return counts

    def count_conflicts(diff_counts):
        """Count number of diffs that appear more than once."""
        conflicts = 0
        for d, c in diff_counts.items():
            if c > 1:
                conflicts += 1
        return conflicts

    # Initialize with best_rows
    current_gaps = rows_to_gaps(best_rows)
    current_all_diffs = compute_all_diffs(current_gaps)
    current_diff_counts = build_diff_counts(current_all_diffs)
    current_conflicts = count_conflicts(current_diff_counts)
    current_scope = max(max(r) for r in best_rows)
    current_cost = current_scope + 10000 * current_conflicts

    best_cost = current_cost
    best_gaps = [list(g) for g in current_gaps]
    best_scope_sa = current_scope

    # Simulated annealing on gap representation
    T = 10.0
    T_min = 0.05
    cooling = 0.9998

    steps = 0
    no_improve_steps = 0
    max_no_improve = 3000

    while time.time() < deadline - 10:
        if T < T_min:
            T = 5.0
            cooling = 0.9997
            if steps > 200000:
                break

        # Pick a random gap to modify
        r_idx = random.randint(0, n_rows - 1)
        g_idx = random.randint(0, k - 1)

        old_gap = current_gaps[r_idx][g_idx]

        # Remove old gap's contribution
        # Recompute all diffs for this row after removing old gap
        # First, compute row from current gaps
        def row_from_gaps(g):
            row = [0]
            for gap in g:
                row.append(row[-1] + gap)
            return row

        old_row = row_from_gaps(current_gaps[r_idx])
        old_diffs = []
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                old_diffs.append(old_row[j] - old_row[i])

        # Remove old diffs from counts
        for d in old_diffs:
            if d in current_diff_counts:
                current_diff_counts[d] -= 1
                if current_diff_counts[d] == 0:
                    del current_diff_counts[d]

        # Try new gap value
        new_gap = None
        accepted = False

        if random.random() < 0.6:
            # Small perturbation
            delta = random.randint(-3, 3)
            new_gap = old_gap + delta
            if new_gap < 1:
                new_gap = 1
        else:
            # Larger perturbation
            max_search = min(best_scope_sa + 20, 300)
            new_gap = random.randint(1, max_search)

        if new_gap == old_gap:
            # Revert
            for d in old_diffs:
                if d in current_diff_counts:
                    current_diff_counts[d] += 1
                else:
                    current_diff_counts[d] = 1
            continue

        # Compute new row with new gap
        current_gaps[r_idx][g_idx] = new_gap
        new_row = row_from_gaps(current_gaps[r_idx])
        new_diffs = []
        for i in range(len(new_row)):
            for j in range(i+1, len(new_row)):
                new_diffs.append(new_row[j] - new_row[i])

        # Check if new diffs are valid (no conflicts with existing)
        valid = True
        for d in new_diffs:
            if d in current_diff_counts and current_diff_counts[d] > 0:
                valid = False
                break

        if valid:
            # Add new diffs to counts
            for d in new_diffs:
                if d in current_diff_counts:
                    current_diff_counts[d] += 1
                else:
                    current_diff_counts[d] = 1

            new_scope = max(max(r) for r in gaps_to_rows(current_gaps))
            new_conflicts = count_conflicts(current_diff_counts)
            new_cost = new_scope + 10000 * new_conflicts

            delta_cost = new_cost - current_cost

            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                current_conflicts = new_conflicts
                current_scope = new_scope
                current_cost = new_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_gaps = [list(g) for g in current_gaps]
                    best_scope_sa = current_scope
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                accepted = True
            else:
                # Revert
                for d in new_diffs:
                    if d in current_diff_counts:
                        current_diff_counts[d] -= 1
                        if current_diff_counts[d] == 0:
                            del current_diff_counts[d]
        else:
            # Revert
            current_gaps[r_idx][g_idx] = old_gap
            for d in old_diffs:
                if d in current_diff_counts:
                    current_diff_counts[d] += 1
                else:
                    current_diff_counts[d] = 1
            continue

        T *= cooling
        steps += 1

        # If stuck, try a fresh greedy restart
        if no_improve_steps > max_no_improve:
            no_improve_steps = 0
            for s in range(10):
                if time.time() > deadline - 20:
                    break
                rows, scope = greedy_construct(random.randint(0, 10000))
                if rows and scope < best_scope_sa:
                    best_scope_sa = scope
                    best_gaps = rows_to_gaps(rows)
                    current_gaps = [list(g) for g in best_gaps]
                    current_all_diffs = compute_all_diffs(current_gaps)
                    current_diff_counts = build_diff_counts(current_all_diffs)
                    current_conflicts = count_conflicts(current_diff_counts)
                    current_scope = best_scope_sa
                    current_cost = current_scope + 10000 * current_conflicts
                    best_cost = current_cost
                    break

    # Convert best_gaps back to rows
    final_rows = gaps_to_rows(best_gaps)

    # Final compaction: try to reduce each element to a smaller value
    used_set = get_used_diffs(final_rows)
    final_scope = max(max(r) for r in final_rows)

    for _ in range(300):
        if time.time() > deadline - 3:
            break
        row_idx = random.randint(0, n_rows - 1)
        row = final_rows[row_idx]
        if len(row) < 2:
            continue
        last = row[-1]
        if last <= 1:
            continue

        # Try to replace the last element with a smaller one
        removed = row.pop()
        for a in row:
            used_set.discard(removed - a)

        found = False
        for x in range(last - 1, 0, -1):
            if x in row:
                continue
            valid = True
            for a in row:
                d = abs(x - a)
                if d == 0 or d in used_set:
                    valid = False
                    break
            if valid:
                pos = 0
                while pos < len(row) and row[pos] < x:
                    pos += 1
                row.insert(pos, x)
                for a in row:
                    if a != x:
                        used_set.add(abs(x - a))
                found = True
                break

        if not found:
            row.append(removed)
            row.sort()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_set.add(row[j] - row[i])

    final_scope = max(max(r) for r in final_rows)

    # Validate final result
    if is_valid(final_rows) and final_scope < best_scope_sa:
        best_scope_sa = final_scope

    # Final safety check
    if not is_valid(final_rows):
        if is_valid(best_rows):
            final_rows = best_rows
        else:
            # Emergency fallback
            used = set()
            final_rows = []
            for _ in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    x = row[-1] + 1
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used:
                                valid = False
                                break
                        if valid:
                            break
                    row.append(x)
                    used.update({x - a for a in row if a != x})
                final_rows.append(row)

    return final_rows
