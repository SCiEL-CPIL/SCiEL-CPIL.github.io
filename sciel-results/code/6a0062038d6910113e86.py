import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
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
            rows.append(row)
        return rows, used

    rows, used_diffs = greedy_construct()
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    # Simulated Annealing parameters
    T = 10.0
    cool_rate = 0.99
    
    last_improvement_time = start_time
    
    while time.time() - start_time < time_limit:
        # Determine move
        # Bias towards reducing the max
        max_row_idx = -1
        max_val = -1
        for i, r in enumerate(rows):
            if r[-1] > max_val:
                max_val = r[-1]
                max_row_idx = i
                
        if random.random() < 0.5:
            r_idx = max_row_idx
            i_idx = k # Try to reduce the last element
        else:
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, k)
            
        old_val = rows[r_idx][i_idx]
        
        # Determine new value
        min_val = rows[r_idx][i_idx - 1] + 1
        if i_idx < k:
            max_val_bound = rows[r_idx][i_idx + 1] - 1
        else:
            max_val_bound = old_val # We will only consider decreasing if it's the last one?
            # Actually, if we pick i_idx=k randomly, we might want to increase it?
            # No, increasing the max element never helps reduce the global max.
            # But it might allow other elements to be rearranged?
            # Generally, for minimizing max, we should focus on decreasing.
            # So for i_idx=k, we only try new_val < old_val.
            
        if i_idx == k:
            if old_val <= min_val:
                continue
            # Pick new_val < old_val
            # To be more effective, pick close to old_val? Or random?
            # Let's pick random in [min_val, old_val - 1]
            new_val = random.randint(min_val, old_val - 1)
        else:
            if old_val + 1 < max_val_bound + 1: # i.e., old_val < max_val_bound
                # Can move up or down
                # Let's pick random in [min_val, max_val_bound]
                new_val = random.randint(min_val, max_val_bound)
            else:
                # Can only move down
                if old_val <= min_val:
                    continue
                new_val = random.randint(min_val, old_val - 1)
                
        if new_val == old_val:
            continue
            
        # Check validity
        old_diffs_to_remove = set()
        for j in range(i_idx):
            old_diffs_to_remove.add(old_val - rows[r_idx][j])
        for j in range(i_idx + 1, k + 1):
            old_diffs_to_remove.add(rows[r_idx][j] - old_val)
            
        new_diffs_to_add = set()
        for j in range(i_idx):
            new_diffs_to_add.add(new_val - rows[r_idx][j])
        for j in range(i_idx + 1, k + 1):
            new_diffs_to_add.add(rows[r_idx][j] - new_val)
            
        # Check conflicts
        # New diffs must not be in (used_diffs - old_diffs_to_remove)
        # Note: used_diffs contains all diffs from all rows.
        # We remove the ones from the old position in this row.
        # Then we check if any new diff is in the remaining used_diffs.
        
        # Optimization: 
        # 1. Remove old_diffs from a temporary set? No, just check intersection.
        # 2. We must also ensure new_diffs_to_add doesn't contain duplicates? 
        #    Since rows[r_idx] is strictly increasing, new_val - rows[r_idx][j] are distinct for j < i_idx.
        #    And rows[r_idx][j] - new_val are distinct for j > i_idx.
        #    And the two sets are disjoint (positive vs negative? No, all diffs are positive magnitudes).
        #    Wait, the task says "differences a_j - a_i (i < j)". These are positive.
        #    So new_diffs_to_add contains positive integers.
        #    Are they distinct within the set?
        #    Yes, because a_j are distinct.
        
        # So we just check:
        # conflict = (new_diffs_to_add & (used_diffs - old_diffs_to_remove))
        
        # To avoid creating a huge set difference every time, we can do:
        # for d in new_diffs_to_add:
        #     if d in used_diffs and d not in old_diffs_to_remove:
        #         conflict = True
        #         break
        
        conflict = False
        for d in new_diffs_to_add:
            if d in used_diffs:
                if d not in old_diffs_to_remove:
                    conflict = True
                    break
                    
        if conflict:
            continue
            
        # Calculate new scope
        # Only the row r_idx changed.
        # New max of row r_idx:
        # Since we changed i_idx, the max is max(rows[r_idx][-1], new_val) if i_idx == k, 
        # or max(rows[r_idx]) otherwise.
        # Actually, just calculate max(rows[r_idx])
        new_row_max = max(rows[r_idx][:i_idx] + [new_val] + rows[r_idx][i_idx+1:])
        # Global max:
        # We need max of all rows.
        # We can store max_per_row.
        # Let's just calculate it for simplicity, it's small.
        current_global_max = max(max(r) for r in rows) # This is old global max, we need new one.
        # New global max = max(new_row_max, max(max(r) for r in rows if r is not rows[r_idx]))
        # Let's precompute max of other rows?
        # For now, just compute.
        
        # Actually, we can update used_diffs and rows, then check scope.
        # But we need to decide acceptance based on scope.
        
        # Let's compute new_scope
        # max_other = max(max(r) for i, r in enumerate(rows) if i != r_idx)
        # new_scope = max(new_row_max, max_other)
        
        # To avoid O(N*K) for max every time, we can maintain a list of row maxes.
        # But N=4, K=8. It's 32 ops. Negligible.
        
        max_other = 0
        for i in range(n_rows):
            if i != r_idx:
                if max(rows[i]) > max_other:
                    max_other = max(rows[i])
        
        new_scope = max(new_row_max, max_other)
        
        # Acceptance
        accepted = False
        if new_scope < best_scope:
            accepted = True
        elif new_scope == best_scope:
            if random.random() < 0.3:
                accepted = True
        else:
            # Simulated Annealing
            delta = new_scope - best_scope
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accepted = True
        
        if accepted:
            # Apply move
            used_diffs.difference_update(old_diffs_to_remove)
            used_diffs.update(new_diffs_to_add)
            rows[r_idx][i_idx] = new_val
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
                last_improvement_time = time.time()
                # Reset temperature? Or just keep cooling.
                # Let's boost T a bit to explore more around the new best.
                T = 10.0 
            else:
                # Cool down
                T *= cool_rate
                
    return best_rows
