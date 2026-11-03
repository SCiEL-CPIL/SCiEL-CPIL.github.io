import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def _compute_all_diffs(rows):
        s = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    s.add(row[j] - row[i])
        return s

    def greedy_construct(seed, strategy='basic'):
        rng = random.Random(seed)
        used = set()
        rows = []

        if strategy == 'random_order':
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            row_slots = [None] * n_rows

            for row_idx in row_order:
                row = [0]
                while len(row) < k + 1:
                    last = row[-1]
                    found = False
                    search_end = last + 40 + rng.randint(0, 30)
                    search_end = min(search_end, lb + 200)
                    candidates = list(range(last + 1, search_end))
                    if rng.random() < 0.5:
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
                        for x in range(last + 1, lb + 300):
                            new_diffs = [x - a for a in row]
                            if any(d in used for d in new_diffs):
                                continue
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                        if not found:
                            return None
                row_slots[row_idx] = row

            if any(r is None for r in row_slots):
                return None
            return row_slots
        else:
            for row_idx in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    last = row[-1]
                    found = False
                    search_end = last + 35 + rng.randint(0, 25)
                    search_end = min(search_end, lb + 200)
                    candidates = list(range(last + 1, search_end))
                    if rng.random() < 0.5:
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
                        for x in range(last + 1, lb + 300):
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

    def local_improve(rows, seed, max_iters=200):
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        used_set = _compute_all_diffs(rows)

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

            for a in row[:elem_idx]:
                used_set.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used_set.discard(a - old_val)
            row.pop(elem_idx)

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

    def random_restart_construct(seed):
        rng = random.Random(seed)
        used = set()
        rows = []

        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                jump = rng.randint(1, 60)
                x = last + jump
                for candidate in range(x - 10, x + 10):
                    if candidate <= last:
                        continue
                    new_diffs = [candidate - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(candidate)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    for candidate in range(last + 1, last + 200):
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

    # Phase 1: Initial greedy constructions
    strategies = ['basic', 'random_order']
    for i in range(3000):
        if time.time() > deadline - 30:
            break
        strategy = strategies[i % len(strategies)]
        rows = greedy_construct(i * 31 + 7, strategy)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    if best_rows is None:
        # Fallback: try more aggressively
        for i in range(2000):
            if time.time() > deadline - 5:
                break
            rows = random_restart_construct(100000 + i * 97)
            if rows is not None:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]

    if best_rows is None:
        return None

    # Phase 2: Local improvement
    improved, imp_scope = local_improve(best_rows, 42, max_iters=500)
    if imp_scope < best_scope:
        best_scope = imp_scope
        best_rows = improved

    # Phase 3: Structural change - difference set partitioning approach
    # Key insight: For a valid row, the set of differences from 0 to each element
    # determines the row. We can try to assign differences to rows more cleverly.
    
    if time.time() < deadline - 20:
        def try_build_row_from_diffs(row_diffs, remaining_diffs, rng, max_val):
            """
            Given a candidate set of row differences, try to build a valid row.
            The row must start at 0, have k+1 elements, and all pairwise diffs in row_diffs.
            Returns the row if successful, None otherwise.
            """
            if len(row_diffs) != k * (k + 1) // 2:
                return None
            
            max_diff = max(row_diffs)
            if max_diff > max_val:
                return None
            
            # The row must end with max_diff
            # Build row: row[0]=0, row[k]=max_diff
            # We need to find row[1], ..., row[k-1]
            
            # Heuristic: the differences from 0 to each element are in row_diffs
            # Let's try to find k elements (including 0 and max_diff) such that
            # all pairwise differences are in row_diffs
            
            # Simple approach: backtracking with pruning
            row = [0]
            used_in_row = set()
            
            def backtrack(current_row, remaining):
                if len(current_row) == k + 1:
                    # Verify all diffs are correct
                    for i in range(len(current_row)):
                        for j in range(i + 1, len(current_row)):
                            d = current_row[j] - current_row[i]
                            if d not in row_diffs:
                                return None
                    return list(current_row)
                
                last = current_row[-1]
                # Try adding elements
                candidates = []
                for d in sorted(remaining):
                    candidate = last + d
                    if candidate > max_val:
                        break
                    # Check if candidate - all existing elements are in row_diffs
                    valid = True
                    for a in current_row:
                        diff = candidate - a
                        if diff not in row_diffs or diff in used_in_row:
                            valid = False
                            break
                    if valid:
                        candidates.append(candidate)
                
                # Limit candidates for speed
                if len(candidates) > 20:
                    rng.shuffle(candidates)
                    candidates = candidates[:20]
                
                for candidate in candidates:
                    new_used = used_in_row | set(candidate - a for a in current_row)
                    result = backtrack(current_row + [candidate], remaining - {candidate - a for a in current_row} - {d for d in remaining if d > candidate})
                    if result is not None:
                        return result
                
                return None
            
            # This backtracking is too slow for large k. Use a simpler heuristic.
            # Instead, try to greedily build the row using the difference set.
            
            # Greedy: start with 0, repeatedly add the smallest valid element
            current_row = [0]
            used_diffs = set()
            
            for step in range(k):
                last = current_row[-1]
                found = False
                # Try candidates in order of the differences
                for d in sorted(row_diffs):
                    candidate = last + d
                    if candidate > max_val:
                        break
                    # Check validity
                    valid = True
                    for a in current_row:
                        diff = candidate - a
                        if diff not in row_diffs or diff in used_diffs:
                            valid = False
                            break
                    if valid:
                        current_row.append(candidate)
                        used_diffs.update(candidate - a for a in current_row[:-1])
                        found = True
                        break
                
                if not found:
                    return None
            
            # Verify
            for i in range(len(current_row)):
                for j in range(i + 1, len(current_row)):
                    d = current_row[j] - current_row[i]
                    if d not in row_diffs:
                        return None
            
            return current_row

        # Try to reconstruct rows using difference set assignment
        # Strategy: assign differences to rows, then try to build each row
        rng = random.Random(12345)
        
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            # Start from current best and try to improve by reassigning differences
            current_rows = [list(r) for r in best_rows]
            current_scope = best_scope
            all_diffs = _compute_all_diffs(current_rows)
            
            # Try to reduce scope by rebuilding one row at a time
            improved_any = False
            for row_idx in range(n_rows):
                if time.time() > deadline - 10:
                    break
                
                # Remove this row's differences
                row_diffs = set()
                row = current_rows[row_idx]
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        row_diffs.add(row[j] - row[i])
                
                remaining_diffs = all_diffs - row_diffs
                
                # Try to build a new row with smaller scope
                target_scope = max(remaining_diffs) if remaining_diffs else 0
                target_scope = min(target_scope, current_scope - 1) if current_scope > lb else lb
                
                # Try several random assignments
                for trial in range(50):
                    if time.time() > deadline - 10:
                        break
                    
                    # Randomly select which differences to assign to this row
                    # Must include differences that form a valid row
                    # Heuristic: pick differences that are "compatible"
                    
                    # Simple approach: try to build a row using a subset of remaining_diffs
                    # that has the right structure
                    
                    # For now, just try to rebuild the row greedily with a target max
                    new_row = [0]
                    used_in_new = set()
                    
                    for step in range(k):
                        last = new_row[-1]
                        found = False
                        for d in range(1, target_scope + 1):
                            candidate = last + d
                            if candidate > target_scope:
                                break
                            valid = True
                            for a in new_row:
                                diff = candidate - a
                                if diff in used_in_new or diff not in all_diffs or diff in row_diffs:
                                    # diff must be in all_diffs but not in other rows
                                    # Actually, diff must be available (not used by other rows)
                                    if diff in remaining_diffs:
                                        continue
                                    valid = False
                                    break
                            if valid:
                                new_row.append(candidate)
                                used_in_new.update(candidate - a for a in new_row[:-1])
                                found = True
                                break
                        
                        if not found:
                            break
                    
                    if len(new_row) == k + 1:
                        # Verify
                        new_row_diffs = set()
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                d = new_row[j] - new_row[i]
                                if d in new_row_diffs:
                                    break
                                new_row_diffs.add(d)
                        else:
                            # Check no conflicts with other rows
                            conflict = False
                            for other_idx in range(n_rows):
                                if other_idx == row_idx:
                                    continue
                                other_diffs = set()
                                other_row = current_rows[other_idx]
                                for i in range(len(other_row)):
                                    for j in range(i + 1, len(other_row)):
                                        other_diffs.add(other_row[j] - other_row[i])
                                if new_row_diffs & other_diffs:
                                    conflict = True
                                    break
                            
                            if not conflict:
                                current_rows[row_idx] = new_row
                                new_scope = get_scope(current_rows)
                                if new_scope < current_scope:
                                    current_scope = new_scope
                                    improved_any = True
                                break
            
            if improved_any and current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]

    # Phase 4: Final local improvement
    if time.time() < deadline - 5:
        improved, imp_scope = local_improve(best_rows, 999, max_iters=300)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    return best_rows


# Quick self-test
if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        # Validate
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
