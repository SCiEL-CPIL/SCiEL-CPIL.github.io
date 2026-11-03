import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Number of differences per row
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    def greedy_construct():
        """Fast greedy construction to get a baseline."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                # Check if x can be added
                valid = True
                for a in row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows

    # Get baseline
    baseline_rows = greedy_construct()
    best_rows = [r[:] for r in baseline_rows]
    best_max = max(max(r) for r in best_rows)
    best_collisions = 0 # Greedy is always valid

    # Helper to compute cost for a list of rows
    def compute_cost(rows):
        used = {}
        max_val = 0
        for r in rows:
            if r[-1] > max_val:
                max_val = r[-1]
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        used[d] += 1
                    else:
                        used[d] = 1
        
        collisions = 0
        for cnt in used.values():
            if cnt > 1:
                collisions += cnt - 1
        
        return collisions, max_val

    # Initial cost
    best_collisions, best_max = compute_cost(best_rows)

    # CSP-based row builder
    def build_row_csp(used_diffs, max_scope, rng, time_limit=0.05):
        """
        Build a single row [0, a1, ..., ak] such that all differences are not in used_diffs.
        Returns the row or None.
        """
        start = time.time()
        # We want to find a row with max element <= max_scope
        # Backtracking search
        row = [0]
        
        def backtrack(idx):
            if time.time() - start > time_limit:
                return False
            if idx == row_len:
                return True
            
            prev_val = row[-1]
            # Heuristic: try values closer to the end to pack tightly, or spread out
            # We need to ensure we have enough room for remaining elements
            remaining = row_len - 1 - idx
            # min_val must be > prev_val
            # max_val must be <= max_scope - remaining (rough estimate)
            
            # Generate candidates
            candidates = []
            min_val = prev_val + 1
            # To ensure we can fit 'remaining' more elements with distinct diffs,
            # we should be careful. For now, just try a range.
            max_val = max_scope - remaining
            
            if min_val > max_val:
                return False
            
            # Randomize order to avoid bias
            # We prioritize smaller values to keep scope low, but shuffle to escape local minima
            # Let's try a mix: some small, some random
            if rng.random() < 0.5:
                candidates = list(range(min_val, max_val + 1))
                rng.shuffle(candidates)
            else:
                # Try to pack tight
                candidates = list(range(min_val, min_val + min(10, max_val - min_val + 1)))
                rng.shuffle(candidates)
                # Also add some random ones
                for _ in range(5):
                    v = rng.randint(min_val, max_val)
                    if v not in candidates:
                        candidates.append(v)

            for v in candidates:
                if v <= prev_val:
                    continue
                # Check differences
                valid = True
                new_diffs = []
                for a in row:
                    d = v - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(v)
                    # Add to used_diffs temporarily? No, we pass a set.
                    # We need to check against global used_diffs AND local row diffs.
                    # The 'used_diffs' passed in is the GLOBAL set.
                    # We must also ensure no collision within the current row.
                    # The loop above checks against 'row' which contains previous elements.
                    # So 'valid' implies no collision with previous elements in this row.
                    # And 'd in used_diffs' implies no collision with other rows.
                    
                    if backtrack(idx + 1):
                        return True
                    
                    row.pop()
            
            return False

        if backtrack(0):
            return row[:]
        return None

    # Main Optimization Loop
    # We use a state that tracks diff counts for efficiency
    
    def run_sa_optimization(initial_rows, time_budget):
        rows = [r[:] for r in initial_rows]
        
        # Build diff counts
        diff_counts = {}
        collisions = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in diff_counts:
                        diff_counts[d] += 1
                        if diff_counts[d] == 2:
                            collisions += 1
                    else:
                        diff_counts[d] = 1
        
        current_collisions = collisions
        current_max = max(max(r) for r in rows)
        
        best_rows_local = [r[:] for r in rows]
        best_collisions_local = current_collisions
        best_max_local = current_max
        
        T = 10.0
        cooling = 0.999
        min_T = 0.05
        
        start = time.time()
        rng = random.Random()
        
        while time.time() - start < time_budget:
            # Choose move type
            # 1. Single element move (fast, fine-grained)
            # 2. Row rebuild (slow, coarse-grained, powerful)
            
            if current_collisions > 0:
                # If invalid, prioritize fixing collisions
                # Row rebuild is very good at this
                move_type = rng.random()
                if move_type < 0.4:
                    move_type = 'rebuild'
                else:
                    move_type = 'single'
            else:
                # If valid, prioritize reducing scope
                move_type = rng.random()
                if move_type < 0.2:
                    move_type = 'rebuild' # Try to find a tighter row
                else:
                    move_type = 'single'
            
            if move_type == 'single':
                r_idx = rng.randint(0, n_rows - 1)
                i_idx = rng.randint(1, row_len - 1)
                
                lower = rows[r_idx][i_idx - 1] + 1
                if i_idx < row_len - 1:
                    upper = rows[r_idx][i_idx + 1] - 1
                else:
                    upper = current_max + 5
                
                if lower > upper:
                    continue
                
                new_val = rng.randint(lower, upper)
                old_val = rows[r_idx][i_idx]
                
                if new_val == old_val:
                    continue
                
                # Remove old diffs
                old_row = rows[r_idx]
                for j in range(row_len):
                    if j != i_idx:
                        d = abs(old_val - old_row[j])
                        diff_counts[d] -= 1
                        if diff_counts[d] == 1:
                            collisions -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                
                # Apply change
                rows[r_idx][i_idx] = new_val
                
                # Add new diffs
                new_row = rows[r_idx]
                for j in range(row_len):
                    if j != i_idx:
                        d = abs(new_val - new_row[j])
                        if d in diff_counts:
                            diff_counts[d] += 1
                            if diff_counts[d] == 2:
                                collisions += 1
                        else:
                            diff_counts[d] = 1
                
                new_collisions = collisions
                new_max = max(rows[r_idx][-1], current_max)
                # Note: if we changed the last element, max might change.
                # For simplicity, recompute max if i_idx == row_len-1 or if we reduced it significantly.
                # Actually, just checking the last element of the modified row and global max is O(N).
                # Let's do a quick check.
                if i_idx == row_len - 1:
                    new_max = max(max(r) for r in rows)
                elif new_val > current_max:
                    new_max = new_val
                else:
                    new_max = current_max

                # Acceptance
                delta_cost = (new_collisions - current_collisions) * 1000 + (new_max - current_max)
                
                if delta_cost < 0 or rng.random() < math.exp(-delta_cost / max(T, 0.01)):
                    current_collisions = new_collisions
                    current_max = new_max
                    
                    if new_collisions < best_collisions_local or (new_collisions == best_collisions_local and new_max < best_max_local):
                        best_collisions_local = new_collisions
                        best_max_local = new_max
                        best_rows_local = [r[:] for r in rows]
                else:
                    # Revert
                    # Remove new diffs
                    for j in range(row_len):
                        if j != i_idx:
                            d = abs(new_val - new_row[j])
                            diff_counts[d] -= 1
                            if diff_counts[d] == 1:
                                collisions -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                    
                    # Restore old val
                    rows[r_idx][i_idx] = old_val
                    
                    # Add old diffs
                    for j in range(row_len):
                        if j != i_idx:
                            d = abs(old_val - old_row[j])
                            if d in diff_counts:
                                diff_counts[d] += 1
                                if diff_counts[d] == 2:
                                    collisions += 1
                            else:
                                diff_counts[d] = 1

            elif move_type == 'rebuild':
                r_idx = rng.randint(0, n_rows - 1)
                old_row = rows[r_idx][:]
                
                # Remove old row diffs
                for i in range(row_len):
                    for j in range(i+1, row_len):
                        d = old_row[j] - old_row[i]
                        diff_counts[d] -= 1
                        if diff_counts[d] == 1:
                            collisions -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                
                used_diffs = set(diff_counts.keys())
                
                # Try to build a new row
                # Target scope: current_max or slightly less
                target_scope = current_max
                # If we are valid, try to reduce scope
                if current_collisions == 0:
                    target_scope = max(1, current_max - rng.randint(1, 5))
                
                new_row = build_row_csp(used_diffs, target_scope, rng, time_limit=0.02)
                
                if new_row is None:
                    # Revert
                    for i in range(row_len):
                        for j in range(i+1, row_len):
                            d = old_row[j] - old_row[i]
                            if d in diff_counts:
                                diff_counts[d] += 1
                                if diff_counts[d] == 2:
                                    collisions += 1
                            else:
                                diff_counts[d] = 1
                    continue
                
                # Add new row diffs
                for i in range(row_len):
                    for j in range(i+1, row_len):
                        d = new_row[j] - new_row[i]
                        if d in diff_counts:
                            diff_counts[d] += 1
                            if diff_counts[d] == 2:
                                collisions += 1
                        else:
                            diff_counts[d] = 1
                
                rows[r_idx] = new_row
                
                new_collisions = collisions
                new_max = max(max(r) for r in rows)
                
                delta_cost = (new_collisions - current_collisions) * 1000 + (new_max - current_max)
                
                if delta_cost < 0 or rng.random() < math.exp(-delta_cost / max(T, 0.01)):
                    current_collisions = new_collisions
                    current_max = new_max
                    
                    if new_collisions < best_collisions_local or (new_collisions == best_collisions_local and new_max < best_max_local):
                        best_collisions_local = new_collisions
                        best_max_local = new_max
                        best_rows_local = [r[:] for r in rows]
                else:
                    # Revert
                    # Remove new
                    for i in range(row_len):
                        for j in range(i+1, row_len):
                            d = new_row[j] - new_row[i]
                            diff_counts[d] -= 1
                            if diff_counts[d] == 1:
                                collisions -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                    
                    rows[r_idx] = old_row
                    
                    # Add old
                    for i in range(row_len):
                        for j in range(i+1, row_len):
                            d = old_row[j] - old_row[i]
                            if d in diff_counts:
                                diff_counts[d] += 1
                                if diff_counts[d] == 2:
                                    collisions += 1
                            else:
                                diff_counts[d] = 1

            T *= cooling
            if T < min_T:
                T = min_T

        return best_rows_local, best_collisions_local, best_max_local

    # Main loop
    while time.time() < deadline:
        # Create initial state
        if random.random() < 0.5:
            init_rows = [r[:] for r in best_rows]
        else:
            # Random initialization
            bound = best_max + 10
            init_rows = []
            for _ in range(n_rows):
                pts = random.sample(range(1, bound + 1), k)
                pts.sort()
                init_rows.append([0] + pts)
        
        # Run SA
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
            
        # Allocate time: 80% to main SA, 20% to polish if valid
        sa_time = remaining * 0.8
        if best_collisions == 0:
            sa_time = remaining * 0.5 # Less time for main if already valid, more for polish?
            # Actually, let's just run main SA. The "rebuild" move with target_scope < current_max acts as polish.
        
        local_rows, local_collisions, local_max = run_sa_optimization(init_rows, sa_time)
        
        if local_collisions < best_collisions or (local_collisions == best_collisions and local_max < best_max):
            best_rows = local_rows
            best_collisions = local_collisions
            best_max = local_max
    
    return best_rows

