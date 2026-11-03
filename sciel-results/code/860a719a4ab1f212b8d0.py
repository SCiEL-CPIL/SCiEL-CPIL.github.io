import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1  # Lower bound on scope

    # Maximum difference we need to track
    max_diff_bound = lb + 400

    def validate_rows(rows):
        """Check all differences are distinct across all rows."""
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

    def bitset_check(used_bits, row, new_val):
        """Check if adding new_val to row creates any duplicate differences.
        used_bits is a set of integers representing which differences are used.
        Returns True if valid, False otherwise.
        """
        for a in row:
            d = new_val - a
            if d in used_bits:
                return False
        return True

    def greedy_with_backtracking(seed, max_scope=None):
        """
        Construct rows using greedy with backtracking.
        Uses bitset-like tracking for fast difference checks.
        """
        rng = random.Random(seed)
        
        if max_scope is None:
            max_scope = lb + 150
        
        used = set()
        rows = []
        
        # Build rows one at a time
        for row_idx in range(n_rows):
            row = [0]
            # Try to build this row
            success = False
            for attempt in range(50):
                row = [0]
                used_snapshot = set(used)
                
                while len(row) < k + 1:
                    last = row[-1]
                    found = False
                    
                    # Determine search range
                    search_end = min(last + 40 + rng.randint(0, 25), max_scope)
                    
                    # Try candidates in order (compact first)
                    for x in range(last + 1, search_end + 1):
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
                        # Failed to complete this row, restore and retry
                        used = used_snapshot
                        row = [0]
                        break
                
                if len(row) == k + 1:
                    success = True
                    break
            
            if not success:
                return None
            
            rows.append(row)
        
        return rows

    def local_compact(rows, seed, max_iters=500):
        """Try to reduce scope by shifting elements left."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Pick a random row and a random non-zero element
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue
            
            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue
            
            # Remove this element
            row.pop(elem_idx)
            
            # Compute all differences in current configuration (without this element)
            used_set = set()
            for i, r in enumerate(rows):
                for a in range(len(r)):
                    for b in range(a + 1, len(r)):
                        used_set.add(r[b] - r[a])
            
            # Try to re-insert at a smaller position
            found = False
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1
            
            for new_val in range(max_val, min_val - 1, -1):
                # Check differences with all elements in this row
                valid = True
                for a in row[:elem_idx]:
                    d = new_val - a
                    if d in used_set:
                        valid = False
                        break
                if not valid:
                    continue
                for a in row[elem_idx:]:
                    d = a - new_val
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    row.insert(elem_idx, new_val)
                    found = True
                    break
            
            if found:
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                # Restore
                row.insert(elem_idx, old_val)
        
        return rows, scope

    def systematic_compact(rows):
        """Systematically try to reduce each element in each row from right to left."""
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        improved = True
        
        while improved:
            if time.time() > deadline - 3:
                break
            improved = False
            
            for row_idx in range(n_rows):
                if time.time() > deadline - 3:
                    break
                row = rows[row_idx]
                # Try from the last element backwards
                for elem_pos in range(len(row) - 1, 0, -1):
                    old_val = row[elem_pos]
                    if old_val <= 1:
                        continue
                    
                    # Remove this element
                    row.pop(elem_pos)
                    
                    # Compute all differences in current configuration
                    used_set = set()
                    for i, r in enumerate(rows):
                        for a in range(len(r)):
                            for b in range(a + 1, len(r)):
                                used_set.add(r[b] - r[a])
                    
                    # Try to re-insert at a smaller position
                    found = False
                    min_val = row[elem_pos - 1] + 1 if elem_pos > 0 else 1
                    max_val = old_val - 1
                    
                    for new_val in range(max_val, min_val - 1, -1):
                        valid = True
                        for a in row[:elem_pos]:
                            d = new_val - a
                            if d in used_set:
                                valid = False
                                break
                        if not valid:
                            continue
                        for a in row[elem_pos:]:
                            d = a - new_val
                            if d in used_set:
                                valid = False
                                break
                        if valid:
                            row.insert(elem_pos, new_val)
                            found = True
                            break
                    
                    if found:
                        new_scope = get_scope(rows)
                        if new_scope < scope:
                            scope = new_scope
                            improved = True
                    else:
                        # Restore
                        row.insert(elem_pos, old_val)
        
        return rows, scope

    def random_restart(seed):
        """Build with completely random choices."""
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                # Random jump size
                jump = rng.randint(1, 50)
                x = last + jump
                # Try this and nearby
                for candidate in range(max(last + 1, x - 15), x + 15):
                    new_diffs = [candidate - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(candidate)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    # Try wider
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

    def simulated_annealing(rows, seed, max_iters=2000):
        """Simulated annealing to reduce scope."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        best_rows = [list(r) for r in rows]
        best_scope = scope
        
        current_temp = 10.0
        cooling_rate = 0.995
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random move: shift an element in a row
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue
            
            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            
            # Try to move this element to a new position
            # Remove it
            row.pop(elem_idx)
            
            used_set = set()
            for i, r in enumerate(rows):
                for a in range(len(r)):
                    for b in range(a + 1, len(r)):
                        used_set.add(r[b] - r[a])
            
            # Try inserting at a new position
            new_pos = rng.randint(0, len(row))
            min_val = row[new_pos - 1] + 1 if new_pos > 0 else 1
            max_val = row[new_pos] - 1 if new_pos < len(row) else old_val + 20
            
            found = False
            for new_val in range(max_val, min_val - 1, -1):
                valid = True
                for a in row[:new_pos]:
                    d = new_val - a
                    if d in used_set:
                        valid = False
                        break
                if not valid:
                    continue
                for a in row[new_pos:]:
                    d = a - new_val
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    row.insert(new_pos, new_val)
                    found = True
                    break
            
            if not found:
                # Restore
                row.insert(elem_idx, old_val)
                current_temp *= cooling_rate
                continue
            
            new_scope = get_scope(rows)
            delta = new_scope - scope
            
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(current_temp, 0.1)):
                scope = new_scope
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
            else:
                # Restore
                row.pop(new_pos)
                row.insert(elem_idx, old_val)
            
            current_temp *= cooling_rate
        
        return best_rows, best_scope

    best_rows = None
    best_scope = float('inf')

    # Phase 1: Greedy constructions with various seeds
    for i in range(2000):
        if time.time() > deadline - 10:
            break
        rows = greedy_with_backtracking(i * 17 + 3)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # Phase 2: Local compaction on best found
    if best_rows is not None:
        improved, imp_scope = local_compact(best_rows, 42, max_iters=500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Phase 3: Random restarts
    for i in range(500):
        if time.time() > deadline - 10:
            break
        rows = random_restart(100000 + i * 97)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # Phase 4: Systematic compaction
    if best_rows is not None:
        improved, imp_scope = systematic_compact(best_rows)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Phase 5: Simulated annealing
    if best_rows is not None:
        improved, imp_scope = simulated_annealing(best_rows, 99999, max_iters=1500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Phase 6: Final systematic compaction
    if best_rows is not None:
        improved, imp_scope = systematic_compact(best_rows)
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
