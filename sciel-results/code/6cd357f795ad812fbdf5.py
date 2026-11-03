import time
import random
import numpy as np

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 115.0  # Safety margin under 120s
    row_len = k + 1
    n_pairs = row_len * (row_len - 1) // 2
    
    # Precompute pair indices for faster diff calculation
    pair_indices = [(i, j) for i in range(row_len) for j in range(i + 1, row_len)]
    
    def row_diffs(row):
        """Return list of all positive differences in a row."""
        return [row[j] - row[i] for i, j in pair_indices]
    
    def max_entry(rows):
        return max(max(r) for r in rows) if rows else 0
    
    def sum_entries(rows):
        return sum(sum(r) for r in rows)
    
    def greedy_row(used, rng, max_val=None):
        """Build one row greedily, choosing the smallest available next mark."""
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            if max_val is not None and x > max_val:
                break
            # Occasionally jump ahead to create diversity
            if rng.random() < 0.15:
                x += rng.randint(1, 5)
                if max_val is not None and x > max_val:
                    break
            
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        return row
    
    def greedy_full(n_rows, rng, max_val=None):
        """Build all rows greedily with randomization."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, rng, max_val)
            if len(row) < row_len:
                return None
            rows.append(row)
        return rows
    
    # Initialize with multiple greedy restarts, keeping the best
    rng = random.Random(42)
    best_matrix = None
    best_scope = float('inf')
    
    for restart in range(15):
        candidate = greedy_full(n_rows, rng)
        if candidate is None:
            continue
        scope = max_entry(candidate)
        if scope < best_scope:
            best_scope = scope
            best_matrix = [list(r) for r in candidate]
        if time.time() > t0 + 5.0:
            break
    
    if best_matrix is None:
        # Fallback: pure greedy without randomization
        used = set()
        best_matrix = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            best_matrix.append(row)
        best_scope = max_entry(best_matrix)
    
    # SA with delta-based conflict tracking for O(1) cost evaluation
    current_rows = [list(r) for r in best_matrix]
    
    # Initialize global diff counts and conflict count
    diff_counts = {}
    conflict_count = 0
    for row in current_rows:
        for d in row_diffs(row):
            old_cnt = diff_counts.get(d, 0)
            if old_cnt >= 1:
                conflict_count += 1
            diff_counts[d] = old_cnt + 1
    
    current_scope = max_entry(current_rows)
    current_sum = sum_entries(current_rows)
    
    PENALTY = 500
    SUM_WEIGHT = 0.05
    current_cost = current_scope + PENALTY * conflict_count + SUM_WEIGHT * current_sum
    
    best_scope = current_scope
    best_conflicts = conflict_count
    best_sum = current_sum
    best_cost = current_cost
    best_rows = [list(r) for r in current_rows]
    
    temp = 30.0
    cooling = 0.9991
    
    row_diffs_list = [row_diffs(row) for row in current_rows]
    
    while time.time() < DEADLINE:
        move_type = rng.random()
        
        if move_type < 0.25:
            # Move: Rebuild a random row (greedy with constraints)
            ri = rng.randrange(n_rows)
            old_row = current_rows[ri]
            old_diffs = row_diffs_list[ri]
            
            # Remove old diffs from global counts
            for d in old_diffs:
                old_cnt = diff_counts[d]
                if old_cnt >= 2:
                    conflict_count -= 1
                if old_cnt == 1:
                    del diff_counts[d]
                else:
                    diff_counts[d] = old_cnt - 1
            
            used = set(diff_counts.keys())
            max_limit = current_scope + rng.randint(0, 8)
            new_row = greedy_row(used, rng, max_limit)
            
            if len(new_row) < row_len:
                # Revert: put old diffs back
                for d in old_diffs:
                    old_cnt = diff_counts.get(d, 0)
                    if old_cnt >= 1:
                        conflict_count += 1
                    diff_counts[d] = old_cnt + 1
                continue
            
            # Add new diffs to global counts
            new_diffs = row_diffs(new_row)
            for d in new_diffs:
                old_cnt = diff_counts.get(d, 0)
                if old_cnt >= 1:
                    conflict_count += 1
                diff_counts[d] = old_cnt + 1
            
            current_rows[ri] = new_row
            row_diffs_list[ri] = new_diffs
            
        elif move_type < 0.75:
            # Move: Shift a single entry in a random row
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            if row_len < 2:
                continue
            
            idx = rng.randint(1, row_len - 1)
            old_val = row[idx]
            
            # More aggressive compression: try larger reductions
            r = rng.random()
            if r < 0.55:
                # Try to reduce (compress)
                new_val = old_val - rng.randint(1, 8)
            elif r < 0.80:
                # Small random perturbation
                new_val = old_val + rng.randint(-5, 5)
            else:
                # Larger random perturbation
                new_val = old_val + rng.randint(-12, 12)
            
            if new_val <= 0 or new_val == old_val:
                continue
            if idx > 1 and new_val <= row[idx - 1]:
                continue
            if idx < row_len - 1 and new_val >= row[idx + 1]:
                continue
            
            old_diffs = row_diffs_list[ri]
            old_diff_set = set(old_diffs)
            
            new_row = list(row)
            new_row[idx] = new_val
            new_diffs = row_diffs(new_row)
            new_diff_set = set(new_diffs)
            
            if len(new_diffs) != len(new_diff_set):
                continue
            
            removed = old_diff_set - new_diff_set
            added = new_diff_set - old_diff_set
            
            delta_conflict = 0
            for d in removed:
                cnt = diff_counts.get(d, 0)
                if cnt >= 2:
                    delta_conflict -= 1
            for d in added:
                cnt = diff_counts.get(d, 0)
                if cnt >= 1:
                    delta_conflict += 1
            
            other_max = 0
            for i in range(n_rows):
                if i != ri:
                    other_max = max(other_max, current_rows[i][-1])
            new_scope = max(new_row[-1], other_max)
            
            delta_scope = new_scope - current_scope
            delta_sum = sum(new_row) - sum(row)
            delta_cost = delta_scope + PENALTY * delta_conflict + SUM_WEIGHT * delta_sum
            
            accept = False
            if delta_cost < 0:
                accept = True
            else:
                if temp > 0.01:
                    prob = np.exp(-delta_cost / temp)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                # Apply the move
                for d in removed:
                    cnt = diff_counts.get(d, 0)
                    if cnt >= 2:
                        conflict_count -= 1
                    if cnt == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = cnt - 1
                for d in added:
                    cnt = diff_counts.get(d, 0)
                    if cnt >= 1:
                        conflict_count += 1
                    diff_counts[d] = cnt + 1
                
                current_rows[ri] = new_row
                row_diffs_list[ri] = new_diffs
                current_scope = new_scope
                current_sum += delta_sum
                current_cost += delta_cost
            # else: reject, do nothing
            
        else:
            # Move: Try to reduce the last element of a row (scope reduction)
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            old_val = row[-1]
            
            # Try to reduce by a small amount
            reduction = rng.randint(1, 6)
            new_val = old_val - reduction
            
            if new_val <= row[-2]:
                continue
            
            old_diffs = row_diffs_list[ri]
            old_diff_set = set(old_diffs)
            
            new_row = list(row)
            new_row[-1] = new_val
            new_diffs = row_diffs(new_row)
            new_diff_set = set(new_diffs)
            
            if len(new_diffs) != len(new_diff_set):
                continue
            
            removed = old_diff_set - new_diff_set
            added = new_diff_set - old_diff_set
            
            delta_conflict = 0
            for d in removed:
                cnt = diff_counts.get(d, 0)
                if cnt >= 2:
                    delta_conflict -= 1
            for d in added:
                cnt = diff_counts.get(d, 0)
                if cnt >= 1:
                    delta_conflict += 1
            
            other_max = 0
            for i in range(n_rows):
                if i != ri:
                    other_max = max(other_max, current_rows[i][-1])
            new_scope = max(new_val, other_max)
            
            delta_scope = new_scope - current_scope
            delta_sum = sum(new_row) - sum(row)
            delta_cost = delta_scope + PENALTY * delta_conflict + SUM_WEIGHT * delta_sum
            
            accept = False
            if delta_cost < 0:
                accept = True
            else:
                if temp > 0.01:
                    prob = np.exp(-delta_cost / temp)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                for d in removed:
                    cnt = diff_counts.get(d, 0)
                    if cnt >= 2:
                        conflict_count -= 1
                    if cnt == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = cnt - 1
                for d in added:
                    cnt = diff_counts.get(d, 0)
                    if cnt >= 1:
                        conflict_count += 1
                    diff_counts[d] = cnt + 1
                
                current_rows[ri] = new_row
                row_diffs_list[ri] = new_diffs
                current_scope = new_scope
                current_sum += delta_sum
                current_cost += delta_cost
        
        # Update best if current is valid and better
        if conflict_count == 0 and current_scope < best_scope:
            best_scope = current_scope
            best_conflicts = conflict_count
            best_sum = current_sum
            best_cost = current_cost
            best_rows = [list(r) for r in current_rows]
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1
    
    # Final polish: try to reduce scope further with targeted moves
    # Use remaining time for a more focused search
    polish_deadline = DEADLINE - 2.0
    
    while time.time() < polish_deadline:
        # Focus on reducing the scope
        ri = rng.randrange(n_rows)
        row = current_rows[ri]
        
        # Try to reduce entries from the end
        for idx in range(row_len - 1, 0, -1):
            old_val = row[idx]
            for reduction in range(1, 7):
                new_val = old_val - reduction
                if new_val <= row[idx - 1]:
                    break
                if idx < row_len - 1 and new_val >= row[idx + 1]:
                    continue
                
                old_diffs = row_diffs_list[ri]
                old_diff_set = set(old_diffs)
                
                new_row = list(row)
                new_row[idx] = new_val
                new_diffs = row_diffs(new_row)
                new_diff_set = set(new_diffs)
                
                if len(new_diffs) != len(new_diff_set):
                    continue
                
                removed = old_diff_set - new_diff_set
                added = new_diff_set - old_diff_set
                
                delta_conflict = 0
                for d in removed:
                    cnt = diff_counts.get(d, 0)
                    if cnt >= 2:
                        delta_conflict -= 1
                for d in added:
                    cnt = diff_counts.get(d, 0)
                    if cnt >= 1:
                        delta_conflict += 1
                
                if delta_conflict > 0:
                    continue  # Don't accept if it creates conflicts
                
                other_max = 0
                for i in range(n_rows):
                    if i != ri:
                        other_max = max(other_max, current_rows[i][-1])
                new_scope = max(new_row[-1], other_max)
                
                if new_scope >= current_scope:
                    continue
                
                # Accept the improvement
                for d in removed:
                    cnt = diff_counts.get(d, 0)
                    if cnt >= 2:
                        conflict_count -= 1
                    if cnt == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = cnt - 1
                for d in added:
                    cnt = diff_counts.get(d, 0)
                    if cnt >= 1:
                        conflict_count += 1
                    diff_counts[d] = cnt + 1
                
                current_rows[ri] = new_row
                row_diffs_list[ri] = new_diffs
                current_scope = new_scope
                current_sum = sum_entries(current_rows)
                
                if conflict_count == 0 and current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
                break  # Move to next idx or row
            
            if time.time() > polish_deadline:
                break
        
        if time.time() > polish_deadline:
            break
    
    # Return the best valid solution found
    # Ensure best_rows is valid
    if best_conflicts == 0:
        return best_rows
    else:
        # Fallback to initial greedy if SA failed
        return best_matrix
