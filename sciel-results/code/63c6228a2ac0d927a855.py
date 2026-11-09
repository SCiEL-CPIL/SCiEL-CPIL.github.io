"""Combined approach: fast bitset-based greedy construction with randomized restarts
and local hill-climbing repair to minimize the scope (largest entry) for
difference triangle set (4,7)."""
import time
import random
import numpy as np


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    total_diffs_needed = n_rows * k * (k + 1) // 2

    def compute_scope(rows):
        return max(max(r) for r in rows) if rows else 0

    def is_valid(rows, k):
        seen = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_build_row(used_bits, k, max_val, row_start=None):
        """Build a row greedily using bitset for O(1) conflict checks."""
        row = list(row_start) if row_start else [0]
        n_marks = len(row)
        while n_marks < k + 1:
            found = False
            for x in range(row[-1] + 1, max_val + 1):
                conflict = False
                for a in row:
                    d = x - a
                    if d > max_val or used_bits[d]:
                        conflict = True
                        break
                if not conflict:
                    for a in row:
                        used_bits[x - a] = True
                    row.append(x)
                    n_marks += 1
                    found = True
                    break
            if not found:
                return None
        return row

    def greedy_construct(n_rows, k, max_val, rng):
        """Build all rows greedily within max_val scope."""
        used = np.zeros(max_val + 1, dtype=bool)
        rows = []
        order = list(range(n_rows))
        rng.shuffle(order)
        for _ in order:
            row = greedy_build_row(used, k, max_val)
            if row is None:
                return None
            rows.append(row)
        return rows

    # Phase 1: Find a valid solution with standard greedy
    rng = random.Random(42)
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)

    best_rows = [r[:] for r in rows]
    best_scope = compute_scope(best_rows)

    # Phase 2: Try to improve scope using randomized greedy with tighter bounds
    # Start from best_scope and try to reduce
    for scope_try in range(best_scope, 100, -1):
        if time.time() > deadline:
            break
        # Try multiple random orderings
        for attempt in range(20):
            if time.time() > deadline:
                break
            result = greedy_construct(n_rows, k, scope_try, rng)
            if result is not None:
                if is_valid(result, k):
                    best_rows = [r[:] for r in result]
                    best_scope = scope_try
                    break  # Found valid solution at this scope, try smaller
        if best_scope == scope_try:
            # Found at this scope, continue to try smaller
            continue
        # If we didn't find at this scope, we can stop trying smaller
        # (since if we can't fit in scope_try, we likely can't fit in smaller)
        # But actually we should keep trying a few more times
        if scope_try < best_scope:
            break

    # Phase 3: Local search / hill climbing to reduce scope
    if time.time() < deadline:
        rows_local = [list(r) for r in best_rows]
        local_best_scope = compute_scope(rows_local)
        
        # Try to move individual marks to smaller positions
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for r_idx in range(n_rows):
                row = rows_local[r_idx]
                for pos in range(k, 0, -1):  # Try from last mark backwards
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    # Try to move this mark to a smaller value
                    for new_val in range(row[pos - 1] + 1, old_val):
                        # Check if moving is valid
                        tent = list(row)
                        tent[pos] = new_val
                        # Check strictly increasing
                        valid = True
                        for i in range(len(tent) - 1):
                            if tent[i + 1] <= tent[i]:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Compute differences of tentative row
                        tent_diffs = set()
                        n = len(tent)
                        for i in range(n):
                            for j in range(i + 1, n):
                                tent_diffs.add(tent[j] - tent[i])
                        
                        # Check no conflict with other rows
                        other_diffs = set()
                        for other_idx in range(n_rows):
                            if other_idx == r_idx:
                                continue
                            other_row = rows_local[other_idx]
                            n_other = len(other_row)
                            for i in range(n_other):
                                for j in range(i + 1, n_other):
                                    other_diffs.add(other_row[j] - other_row[i])
                        
                        if tent_diffs & other_diffs:
                            continue
                        
                        # Accept the move
                        rows_local[r_idx] = tent
                        new_scope = compute_scope(rows_local)
                        if new_scope < local_best_scope:
                            local_best_scope = new_scope
                            improved = True
                        break  # Move to next position
            
            if local_best_scope < best_scope:
                best_rows = [list(r) for r in rows_local]
                best_scope = local_best_scope

    # Final validation
    if not is_valid(best_rows, k):
        # Fallback to original greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return best_rows
