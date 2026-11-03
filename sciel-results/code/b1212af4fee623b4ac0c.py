import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1

    best_rows = None
    best_scope = float('inf')

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

    def greedy_construct(seed, strategy='basic'):
        """
        Construct rows using greedy with randomization.
        strategy: 'basic', 'random_order', 'compact', 'spread'
        """
        rng = random.Random(seed)
        used = set()
        rows = []

        if strategy == 'random_order':
            # Decide row "targets" first, then build
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            row_slots = [None] * n_rows

            for row_idx in row_order:
                row = [0]
                while len(row) < k + 1:
                    last = row[-1]
                    found = False
                    # Try a range of candidates
                    # Bias toward compact
                    if strategy == 'compact':
                        search_end = last + 30 + rng.randint(0, 20)
                    elif strategy == 'spread':
                        search_end = last + 50 + rng.randint(0, 40)
                    else:
                        search_end = last + 40 + rng.randint(0, 30)

                    # Try candidates in slightly randomized order
                    candidates = list(range(last + 1, min(search_end, lb + 200)))
                    # Bias: mostly try smaller first
                    if rng.random() < 0.6:
                        # Keep order but perturb
                        for i in range(len(candidates) - 1, 0, -1):
                            if rng.random() < 0.1:
                                candidates[i], candidates[i-1] = candidates[i-1], candidates[i]
                    else:
                        rng.shuffle(candidates)

                    for x in candidates:
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        # Valid
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break

                    if not found:
                        # Wider search
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
            # Build rows in order
            for row_idx in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    last = row[-1]
                    found = False

                    if strategy == 'compact':
                        search_end = last + 25 + rng.randint(0, 15)
                    elif strategy == 'spread':
                        search_end = last + 50 + rng.randint(0, 40)
                    else:
                        search_end = last + 35 + rng.randint(0, 25)

                    candidates = list(range(last + 1, min(search_end, lb + 200)))
                    if rng.random() < 0.5:
                        # Mostly in order
                        pass
                    else:
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
        """Try to reduce scope by shifting elements left."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)

        for _ in range(max_iters):
            if time.time() > deadline - 2:
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

            # Try to shift this element (and all after it) left
            # First, compute all differences involving this element
            # Actually, let's try to reduce the last element first
            # Remove element at elem_idx and all after, then re-add

            # Simpler: try to reduce the last element of a random row
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue
            last_idx = len(row) - 1
            last_val = row[last_idx]
            if last_val <= 1:
                continue

            # Try new values for last element
            # Remove last element's differences from consideration
            # Compute which differences are "owned" by this element
            # For simplicity: remove element, try new values, check against all other diffs

            # Remove the last element
            removed = row.pop()

            # Compute all differences in remaining rows + this row
            # We need to check new value against all existing differences
            # and all differences within this row
            used_set = set()
            for i, r in enumerate(rows):
                if i == row_idx:
                    # This row without the last element
                    for a in range(len(r)):
                        for b in range(a + 1, len(r)):
                            used_set.add(r[b] - r[a])
                else:
                    for a in range(len(r)):
                        for b in range(a + 1, len(r)):
                            used_set.add(r[b] - r[a])

            # Also need differences from new element to all in this row
            # Try new values
            found = False
            for new_val in range(removed - 1, row[-1] if len(row) > 0 else 0, -1):
                if new_val <= row[-1]:
                    continue
                # Check new differences
                valid = True
                new_diffs = [new_val - a for a in row]
                for d in new_diffs:
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    # Check new_diffs against each other (they're all positive, distinct since row is sorted)
                    row.append(new_val)
                    # Update used_set implicitly - we just need to check against global
                    # Actually we need to ensure new_diffs don't conflict with each other (they don't, row is sorted)
                    found = True
                    break

            if found:
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                # Restore
                row.append(removed)

        return rows, scope

    def random_restart_construct(seed):
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
                jump = rng.randint(1, 60)
                x = last + jump
                # Try this and nearby
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

    def simulated_annealing(initial_rows, seed, max_iters=5000):
        """
        Refine rows using Simulated Annealing.
        Moves:
        1. Replace an element in a row with a different valid value.
        2. Swap two elements between two rows (if valid).
        """
        rng = random.Random(seed)
        rows = [list(r) for r in initial_rows]
        current_scope = get_scope(rows)
        best_rows = [list(r) for r in rows]
        best_scope = current_scope
        
        def compute_all_diffs(rs):
            diffs = set()
            for r in rs:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        diffs.add(r[j]-r[i])
            return diffs
        
        current_diffs = compute_all_diffs(rows)
        
        for i in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Pick a random move
            move_type = rng.randint(0, 1)
            
            if move_type == 0:
                # Replace element in a row
                r_idx = rng.randint(0, n_rows-1)
                e_idx = rng.randint(1, len(rows[r_idx])-1)
                old_val = rows[r_idx][e_idx]
                
                # Try to find a new value for rows[r_idx][e_idx]
                # Remove old diffs involving this element
                row = rows[r_idx]
                old_diffs = set()
                for j in range(len(row)):
                    if j == e_idx: continue
                    if j < e_idx:
                        old_diffs.add(old_val - row[j])
                    else:
                        old_diffs.add(row[j] - old_val)
                
                # Candidate new values
                # Range: between prev and next
                prev_val = row[e_idx-1]
                next_val = row[e_idx+1] if e_idx+1 < len(row) else float('inf')
                
                # Try a few random candidates in the range
                found = False
                candidates = []
                # Bias towards smaller values to reduce scope
                search_range = range(prev_val + 1, min(next_val, old_val + 10))
                if len(search_range) > 0:
                    candidates = list(search_range)
                    rng.shuffle(candidates)
                
                for new_val in candidates:
                    if new_val == old_val: continue
                    if new_val <= prev_val or (e_idx+1 < len(row) and new_val >= next_val): continue
                    
                    # Calculate new diffs
                    new_diffs = set()
                    for j in range(len(row)):
                        if j == e_idx: continue
                        if j < e_idx:
                            new_diffs.add(new_val - row[j])
                        else:
                            new_diffs.add(row[j] - new_val)
                    
                    # Check validity
                    valid = True
                    for d in new_diffs:
                        if d in current_diffs and d not in old_diffs:
                            valid = False
                            break
                    
                    if valid:
                        # Apply move
                        rows[r_idx][e_idx] = new_val
                        current_diffs = (current_diffs - old_diffs) | new_diffs
                        found = True
                        break
                
                if not found:
                    continue

            else:
                # Swap elements between two rows
                r1 = rng.randint(0, n_rows-1)
                r2 = rng.randint(0, n_rows-1)
                if r1 == r2: continue
                
                e1 = rng.randint(1, len(rows[r1])-1)
                e2 = rng.randint(1, len(rows[r2])-1)
                
                v1 = rows[r1][e1]
                v2 = rows[r2][e2]
                
                # Check if swapping v1 and v2 maintains sorted order
                if e1 > 0 and v2 <= rows[r1][e1-1]: continue
                if e1+1 < len(rows[r1]) and v2 >= rows[r1][e1+1]: continue
                if e2 > 0 and v1 <= rows[r2][e2-1]: continue
                if e2+1 < len(rows[r2]) and v1 >= rows[r2][e2+1]: continue
                
                # Calculate diffs to remove and add
                row1 = rows[r1]
                row2 = rows[r2]
                
                old_diffs1 = set()
                for j in range(len(row1)):
                    if j == e1: continue
                    if j < e1: old_diffs1.add(v1 - row1[j])
                    else: old_diffs1.add(row1[j] - v1)
                    
                old_diffs2 = set()
                for j in range(len(row2)):
                    if j == e2: continue
                    if j < e2: old_diffs2.add(v2 - row2[j])
                    else: old_diffs2.add(row2[j] - v2)
                
                # New diffs
                new_diffs1 = set()
                for j in range(len(row1)):
                    if j == e1: continue
                    if j < e1: new_diffs1.add(v2 - row1[j])
                    else: new_diffs1.add(row1[j] - v2)
                    
                new_diffs2 = set()
                for j in range(len(row2)):
                    if j == e2: continue
                    if j < e2: new_diffs2.add(v1 - row2[j])
                    else: new_diffs2.add(row2[j] - v1)
                
                old_diffs = old_diffs1 | old_diffs2
                new_diffs = new_diffs1 | new_diffs2
                
                valid = True
                for d in new_diffs:
                    if d in current_diffs and d not in old_diffs:
                        valid = False
                        break
                
                if valid:
                    # Apply swap
                    rows[r1][e1] = v2
                    rows[r2][e2] = v1
                    current_diffs = (current_diffs - old_diffs) | new_diffs

            # Evaluate
            new_scope = get_scope(rows)
            
            # Acceptance
            if new_scope < current_scope:
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
            elif new_scope == current_scope:
                # Accept to explore
                pass
            else:
                # Revert
                if move_type == 0:
                    rows[r_idx][e_idx] = old_val
                    current_diffs = (current_diffs - new_diffs) | old_diffs
                else:
                    rows[r1][e1] = v1
                    rows[r2][e2] = v2
                    current_diffs = (current_diffs - new_diffs) | old_diffs
                continue

        return best_rows, best_scope

    def final_polish(rows, seed, max_iters=5000):
        """
        High-precision polish pass: try to reduce scope by 1 at a time.
        Systematically try to move each element to a smaller value.
        """
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        
        def compute_all_diffs(rs):
            diffs = set()
            for r in rs:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        diffs.add(r[j]-r[i])
            return diffs
        
        current_diffs = compute_all_diffs(rows)
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Try to reduce the maximum element
            max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
            row = rows[max_row_idx]
            last_val = row[-1]
            
            if last_val <= 1:
                break
            
            # Try to reduce last_val by 1, 2, 3...
            found = False
            for delta in range(1, 10):
                new_val = last_val - delta
                if new_val <= row[-2]:
                    break
                
                # Remove old diffs for last element
                old_diffs = set()
                for j in range(len(row) - 1):
                    old_diffs.add(last_val - row[j])
                
                # New diffs
                new_diffs = set()
                for j in range(len(row) - 1):
                    new_diffs.add(new_val - row[j])
                
                # Check validity
                valid = True
                for d in new_diffs:
                    if d in current_diffs and d not in old_diffs:
                        valid = False
                        break
                
                if valid:
                    # Apply
                    row[-1] = new_val
                    current_diffs = (current_diffs - old_diffs) | new_diffs
                    scope = get_scope(rows)
                    found = True
                    break
            
            if not found:
                # Try other rows
                for r_idx in range(n_rows):
                    if r_idx == max_row_idx:
                        continue
                    row = rows[r_idx]
                    last_val = row[-1]
                    if last_val <= 1:
                        continue
                    
                    for delta in range(1, 5):
                        new_val = last_val - delta
                        if new_val <= row[-2]:
                            break
                        
                        old_diffs = set()
                        for j in range(len(row) - 1):
                            old_diffs.add(last_val - row[j])
                        
                        new_diffs = set()
                        for j in range(len(row) - 1):
                            new_diffs.add(new_val - row[j])
                        
                        valid = True
                        for d in new_diffs:
                            if d in current_diffs and d not in old_diffs:
                                valid = False
                                break
                        
                        if valid:
                            row[-1] = new_val
                            current_diffs = (current_diffs - old_diffs) | new_diffs
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                # Try moving non-last elements
                r_idx = rng.randint(0, n_rows - 1)
                row = rows[r_idx]
                e_idx = rng.randint(1, len(row) - 2)  # Not last
                old_val = row[e_idx]
                if old_val <= row[e_idx - 1] + 1:
                    continue
                
                prev_val = row[e_idx - 1]
                next_val = row[e_idx + 1]
                
                for delta in range(1, 5):
                    new_val = old_val - delta
                    if new_val <= prev_val:
                        break
                    if new_val >= next_val:
                        continue
                    
                    old_diffs = set()
                    for j in range(len(row)):
                        if j == e_idx: continue
                        if j < e_idx:
                            old_diffs.add(old_val - row[j])
                        else:
                            old_diffs.add(row[j] - old_val)
                    
                    new_diffs = set()
                    for j in range(len(row)):
                        if j == e_idx: continue
                        if j < e_idx:
                            new_diffs.add(new_val - row[j])
                        else:
                            new_diffs.add(row[j] - new_val)
                    
                    valid = True
                    for d in new_diffs:
                        if d in current_diffs and d not in old_diffs:
                            valid = False
                            break
                    
                    if valid:
                        row[e_idx] = new_val
                        current_diffs = (current_diffs - old_diffs) | new_diffs
                        found = True
                        break
        
        return rows, scope

    # Strategy 1: Many greedy attempts with different strategies
    strategies = ['basic', 'compact', 'spread', 'random_order']
    attempt_count = 0
    for i in range(3000):
        if time.time() > deadline - 5:
            break
        strategy = strategies[i % len(strategies)]
        rows = greedy_construct(i * 31 + 7, strategy)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                attempt_count += 1
                if best_scope == lb:
                    break
        attempt_count += 1

    # Strategy 2: Local improvement on best found
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 42, max_iters=500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
        if time.time() > deadline - 5:
            return best_rows

    # Strategy 3: Random restarts
    for i in range(500):
        if time.time() > deadline - 5:
            break
        rows = random_restart_construct(100000 + i * 97)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # Strategy 4: Local improvement on best found again
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 12345, max_iters=300)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Strategy 5: Simulated Annealing / Hill Climbing
    if best_rows is not None:
        # Run SA on the best found so far
        # Use multiple seeds for SA
        for sa_seed in range(10):
            if time.time() > deadline - 5:
                break
            improved, imp_scope = simulated_annealing(best_rows, 50000 + sa_seed * 13, max_iters=2000)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Strategy 6: Final polish pass
    if best_rows is not None:
        improved, imp_scope = final_polish(best_rows, 99999, max_iters=5000)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Final local improvement pass
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 99999, max_iters=1000)
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
