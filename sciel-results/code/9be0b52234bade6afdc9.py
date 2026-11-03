import time
import random
import sys
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    # Total number of positive differences needed
    num_diffs = n_rows * k * (k + 1) // 2
    lb = num_diffs  # counting lower bound (scope must be at least this)
    k1 = k + 1

    best_rows = None
    best_scope = float('inf')

    # --- Fast greedy baseline (smallest valid x each step) ---
    def greedy_baseline():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # --- NumPy-based backtracking with bitset feasibility ---
    def np_backtracking(max_scope, seed, time_limit=2.0):
        rng = random.Random(seed)
        t_start = time.time()
        
        # Bitset for used differences
        bitset = np.zeros(max_scope + 1, dtype=np.uint8)
        rows = []

        def check_and_set(row_so_far, x):
            """Check if x can be added; if yes, mark diffs and return list of diffs set."""
            diffs = np.array([x - a for a in row_so_far])
            if np.any(bitset[diffs]):
                return None
            bitset[diffs] = 1
            return diffs

        def unset(diffs):
            bitset[diffs] = 0

        def build_row(pos, last_val, row_so_far):
            if time.time() - t_start > time_limit:
                return None
            if pos == k1:
                return list(row_so_far)

            # Prune: need (k1 - pos) more elements
            max_x = max_scope - (k1 - pos - 1)
            if max_x <= last_val:
                return None

            # Collect valid x values using numpy for speed
            valid_xs = []
            # Sample a window of candidates (bias toward smaller)
            search_start = last_val + 1
            search_end = min(max_x, last_val + 50)
            
            for x in range(search_start, search_end + 1):
                diffs = np.array([x - a for a in row_so_far])
                if not np.any(bitset[diffs]):
                    valid_xs.append(x)
            
            # If no valid in window, try a few random spots
            if not valid_xs and max_x > last_val + 50:
                for _ in range(5):
                    x = rng.randint(last_val + 1, max_x)
                    diffs = np.array([x - a for a in row_so_far])
                    if not np.any(bitset[diffs]):
                        valid_xs.append(x)
            
            if not valid_xs:
                return None

            rng.shuffle(valid_xs)
            
            for x in valid_xs:
                diffs = np.array([x - a for a in row_so_far])
                bitset[diffs] = 1
                res = build_row(pos + 1, x, row_so_far + [x])
                if res is not None:
                    return res
                bitset[diffs] = 0
            return None

        for _ in range(n_rows):
            row = build_row(1, 0, [0])
            if row is None:
                return None
            rows.append(row)
            # Mark all diffs in this row
            for i in range(k1):
                for j in range(i + 1, k1):
                    bitset[row[j] - row[i]] = 1
        return rows

    # --- Simulated annealing with numpy bitset and difference-first moves ---
    def simulated_annealing(rows, seed, time_limit=15.0):
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = max(max(r) for r in rows)
        
        # Use numpy bitset for O(1) diff lookups
        bitset = np.zeros(scope + 1, dtype=np.uint8)
        for row in rows:
            for i in range(k1):
                for j in range(i + 1, k1):
                    bitset[row[j] - row[i]] = 1
        
        t_start = time.time()
        initial_temp = max(1.0, scope * 0.1)
        current_temp = initial_temp
        cooling_rate = 0.995
        min_temp = 0.01
        
        best_rows = [list(r) for r in rows]
        best_scope = scope
        
        def row_diffs(row):
            """Get all positive differences in a row as numpy array."""
            diffs = []
            for i in range(k1):
                for j in range(i + 1, k1):
                    diffs.append(row[j] - row[i])
            return np.array(diffs)
        
        def can_place(row_without_elem, elem_idx, new_val):
            """Check if new_val can be inserted at elem_idx in row_without_elem."""
            # Diffs with elements before elem_idx
            for a in row_without_elem[:elem_idx]:
                d = new_val - a
                if d > scope or bitset[d]:
                    return False
            # Diffs with elements after elem_idx
            for a in row_without_elem[elem_idx:]:
                d = a - new_val
                if d > scope or bitset[d]:
                    return False
            return True
        
        while current_temp > min_temp and time.time() - t_start < time_limit:
            if time.time() > deadline - 5:
                break
            
            move_type = rng.random()
            
            if move_type < 0.7:
                # Single element shift
                row_idx = rng.randint(0, n_rows - 1)
                row = rows[row_idx]
                elem_idx = rng.randint(1, len(row) - 1)
                old_val = row[elem_idx]
                if old_val <= 1:
                    continue
                
                # Remove element and its diffs
                for a in row[:elem_idx]:
                    bitset[old_val - a] = 0
                for a in row[elem_idx + 1:]:
                    bitset[a - old_val] = 0
                row.pop(elem_idx)
                
                min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
                max_val = min(old_val + rng.randint(-3, 3), scope)
                
                found = False
                for new_val in range(min_val, max_val + 1):
                    if can_place(row, elem_idx, new_val):
                        row.insert(elem_idx, new_val)
                        for a in row[:elem_idx]:
                            bitset[new_val - a] = 1
                        for a in row[elem_idx + 1:]:
                            bitset[a - new_val] = 1
                        found = True
                        break
                
                if not found:
                    row.insert(elem_idx, old_val)
                    for a in row[:elem_idx]:
                        bitset[old_val - a] = 1
                    for a in row[elem_idx + 1:]:
                        bitset[a - old_val] = 1
                    continue
                
                new_scope = max(max(r) for r in rows)
                delta = new_scope - scope
                if delta <= 0 or rng.random() < math.exp(-delta / max(current_temp, 1e-6)):
                    scope = new_scope
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in rows]
                else:
                    # Revert
                    for a in row[:elem_idx]:
                        bitset[new_val - a] = 0
                    for a in row[elem_idx + 1:]:
                        bitset[a - new_val] = 0
                    row.pop(elem_idx)
                    row.insert(elem_idx, old_val)
                    for a in row[:elem_idx]:
                        bitset[old_val - a] = 1
                    for a in row[elem_idx + 1:]:
                        bitset[a - old_val] = 1
            else:
                # Row regeneration with greedy fill
                row_idx = rng.randint(0, n_rows - 1)
                old_row = list(rows[row_idx])
                
                # Remove old row's diffs
                old_diffs = row_diffs(old_row)
                bitset[old_diffs] = 0
                
                # Build new row greedily
                new_row = [0]
                x = 0
                success = True
                while len(new_row) < k1:
                    x += 1
                    if x > scope:
                        success = False
                        break
                    new_diffs = np.array([x - a for a in new_row])
                    if not np.any(bitset[new_diffs]):
                        new_row.append(x)
                        bitset[new_diffs] = 1
                
                if not success:
                    # Restore old row
                    bitset[old_diffs] = 1
                    continue
                
                rows[row_idx] = new_row
                new_scope = max(max(r) for r in rows)
                delta = new_scope - scope
                if delta <= 0 or rng.random() < math.exp(-delta / max(current_temp, 1e-6)):
                    scope = new_scope
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in rows]
                else:
                    # Revert
                    new_diffs = row_diffs(new_row)
                    bitset[new_diffs] = 0
                    rows[row_idx] = old_row
                    bitset[old_diffs] = 1
            
            current_temp *= cooling_rate
        
        return best_rows, best_scope

    # --- Local improvement: shift elements left to reduce scope ---
    def local_improve(rows, seed, max_iters=2000):
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = max(max(r) for r in rows)
        k1 = k + 1

        # Build used set
        used = set()
        for row in rows:
            for i in range(k1):
                for j in range(i + 1, k1):
                    used.add(row[j] - row[i])

        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue

            # Remove element and its diffs
            for a in row[:elem_idx]:
                used.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used.discard(a - old_val)
            row.pop(elem_idx)

            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1
            found = False
            # Try from max_val down to min_val (prefer smallest valid)
            for new_val in range(min_val, max_val + 1):
                valid = True
                for a in row[:elem_idx]:
                    if (new_val - a) in used:
                        valid = False
                        break
                if valid:
                    for a in row[elem_idx:]:
                        if (a - new_val) in used:
                            valid = False
                            break
                if valid:
                    row.insert(elem_idx, new_val)
                    for a in row[:elem_idx]:
                        used.add(new_val - a)
                    for a in row[elem_idx + 1:]:
                        used.add(a - new_val)
                    found = True
                    break

            if found:
                new_scope = max(max(r) for r in rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used.add(a - old_val)

        return rows, scope

    # --- Systematic compaction pass ---
    def compact_pass(rows, seed=0):
        rows = [list(r) for r in rows]
        scope = max(max(r) for r in rows)
        k1 = k + 1
        used = set()
        for row in rows:
            for i in range(k1):
                for j in range(i + 1, k1):
                    used.add(row[j] - row[i])

        improved = True
        rounds = 0
        while improved and time.time() < deadline - 2 and rounds < 200:
            improved = False
            rounds += 1
            for row_idx in range(n_rows):
                row = rows[row_idx]
                for elem_pos in range(len(row) - 1, 0, -1):
                    old_val = row[elem_pos]
                    if old_val <= 1:
                        continue
                    for a in row[:elem_pos]:
                        used.discard(old_val - a)
                    for a in row[elem_pos + 1:]:
                        used.discard(a - old_val)
                    row.pop(elem_pos)
                    min_val = row[elem_pos - 1] + 1 if elem_pos > 0 else 1
                    found = False
                    for new_val in range(min_val, old_val):
                        valid = True
                        for a in row[:elem_pos]:
                            if (new_val - a) in used:
                                valid = False
                                break
                        if valid:
                            for a in row[elem_pos:]:
                                if (a - new_val) in used:
                                    valid = False
                                    break
                        if valid:
                            row.insert(elem_pos, new_val)
                            for a in row[:elem_pos]:
                                used.add(new_val - a)
                            for a in row[elem_pos + 1:]:
                                used.add(a - new_val)
                            found = True
                            break
                    if found:
                        new_scope = max(max(r) for r in rows)
                        if new_scope < scope:
                            scope = new_scope
                            improved = True
                    else:
                        row.insert(elem_pos, old_val)
                        for a in row[:elem_pos]:
                            used.add(old_val - a)
                        for a in row[elem_pos + 1:]:
                            used.add(a - old_val)
        return rows, scope

    # Phase 1: Greedy baseline
    rows = greedy_baseline()
    if rows:
        best_scope = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]

    # Phase 2: Local improvement on greedy
    if best_rows:
        improved, imp_scope = local_improve(best_rows, 42, max_iters=500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
        # Compact pass
        compacted, comp_scope = compact_pass(best_rows)
        if comp_scope < best_scope:
            best_scope = comp_scope
            best_rows = compacted

    # Phase 3: NumPy backtracking search with decreasing target scope
    target = best_scope - 1
    attempt = 0
    while target > lb and time.time() < deadline - 10:
        found = False
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            sol = np_backtracking(target, 100000 + attempt, time_limit=1.5)
            if sol:
                new_scope = max(max(r) for r in sol)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in sol]
                    found = True
                    # Improve
                    improved, imp_scope = local_improve(best_rows, attempt + 500, max_iters=300)
                    if imp_scope < best_scope:
                        best_scope = imp_scope
                        best_rows = improved
                    compacted, comp_scope = compact_pass(best_rows)
                    if comp_scope < best_scope:
                        best_scope = comp_scope
                        best_rows = compacted
                    break
            attempt += 1
        if found:
            target = best_scope - 1
        else:
            target -= 1
            if target <= lb:
                break

    # Phase 4: Simulated annealing
    if best_rows and time.time() < deadline - 15:
        sa_rows, sa_scope = simulated_annealing(best_rows, 88888, time_limit=12.0)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows
        # Polish after SA
        improved, imp_scope = local_improve(best_rows, 99999, max_iters=500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
        compacted, comp_scope = compact_pass(best_rows)
        if comp_scope < best_scope:
            best_scope = comp_scope
            best_rows = compacted

    # Phase 5: More simulated annealing with different seeds
    if best_rows and time.time() < deadline - 10:
        for seed_offset in [11111, 22222, 33333]:
            if time.time() > deadline - 5:
                break
            sa_rows, sa_scope = simulated_annealing(best_rows, 88888 + seed_offset, time_limit=5.0)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows
            improved, imp_scope = local_improve(best_rows, 99999 + seed_offset, max_iters=300)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
            compacted, comp_scope = compact_pass(best_rows)
            if comp_scope < best_scope:
                best_scope = comp_scope
                best_rows = compacted

    # Phase 6: Final high-precision polish
    if best_rows and time.time() < deadline - 3:
        # Multiple rounds of aggressive compaction
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            compacted, comp_scope = compact_pass(best_rows)
            if comp_scope < best_scope:
                best_scope = comp_scope
                best_rows = compacted
            else:
                break

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
