import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Greedy Init
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
        
    best_rows = [r[:] for r in rows]
    best_cost = max(max(r) for r in best_rows) # Since valid, cost is just max
    
    # Precompute diffs
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    current_diffs = [get_diffs(r) for r in rows]
    all_diffs_count = {}
    for d_set in current_diffs:
        for d in d_set:
            all_diffs_count[d] = all_diffs_count.get(d, 0) + 1
            
    def get_conflicts():
        return sum(1 for v in all_diffs_count.values() if v > 1)
        
    def get_max_val():
        return max(max(r) for r in rows)
        
    current_cost = get_max_val() + 1000 * get_conflicts()
    
    T = 100.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick move
        r = random.randint(0, n_rows - 1)
        j = random.randint(1, k) # Index 1 to k
        
        old_v = rows[r][j]
        
        # Determine range
        L = rows[r][j-1] + 1
        if j < k:
            R = rows[r][j+1] - 1
        else:
            # If last element, allow it to grow or shrink
            # To minimize max, we prefer shrinking.
            # Let's sample from [L, old_v + 10]
            R = old_v + 10
            
        if L > R:
            continue
            
        new_v = random.randint(L, R)
        if new_v == old_v:
            continue
            
        # Calculate new diffs for row r
        old_diffs = current_diffs[r]
        # Remove diffs involving j
        # Diffs are (j, i) and (k, j)
        # Actually, easier to recompute row r diffs?
        # Row length is k+1 (9). Diffs are 36. Recomputing is fast.
        
        new_row = rows[r][:]
        new_row[j] = new_v
        
        # Check if strictly increasing
        if new_row[j] <= new_row[j-1] or (j < k and new_row[j] >= new_row[j+1]):
             continue
             
        new_diffs_r = get_diffs(new_row)
        
        # Update global counts
        # Remove old_diffs_r from global
        # Add new_diffs_r to global
        # This is O(K^2) which is small.
        
        # Efficient update:
        # We need to know which diffs were removed and which added.
        removed = old_diffs - new_diffs_r
        added = new_diffs_r - old_diffs
        
        # Check validity of added
        # A diff is valid if its count in (all_diffs_count - removed + added) is 1.
        # Actually, we just need to check if any `d` in `added` is already present in `all_diffs_count` 
        # after removing `removed`.
        
        # Let's just compute the new conflict count roughly.
        # Current conflicts: C
        # For each d in removed:
        #   if count[d] > 1, C decreases? No, if count[d] becomes 1, conflicts decreases by 1.
        #   if count[d] becomes 0, conflicts unchanged (was 1, now 0, 0 conflicts).
        # For each d in added:
        #   if count[d] (after removals) >= 1, then new count >= 2, conflicts increases by 1.
        
        # Let's just recalculate conflicts for simplicity, as K is small.
        # Or, maintain a global dict `diff_count`.
        
        # Let's use a helper to compute cost delta.
        
        # Save state
        rows[r] = new_row
        current_diffs[r] = new_diffs_r
        
        # Update global diff counts
        # This is getting complex to do incrementally without bugs.
        # Given K=8, n_rows=4, total diffs=144.
        # Recalculating all diffs and conflicts is O(144).
        # This is very fast.
        
        # Let's just recalculate.
        all_diffs = set()
        conflicts = 0
        # We can use a Counter
        from collections import Counter
        c = Counter()
        max_val = 0
        for r_idx in range(n_rows):
            for d in current_diffs[r_idx]:
                c[d] += 1
            m = max(rows[r_idx])
            if m > max_val: max_val = m
            
        for count in c.values():
            if count > 1:
                conflicts += 1 # Or sum(count-1)? 
                # "all differences ... are distinct"
                # So if a diff appears twice, it's 1 conflict pair? 
                # The score is based on validity. Invalid = 0 score.
                # So we just need to know if it's valid.
                # Cost = MaxVal + 1000 * (1 if invalid else 0)?
                # No, SA needs a gradient. 
                # Cost = MaxVal + 100 * (number of duplicate diffs).
                
        new_cost = max_val + 100 * sum(v - 1 for v in c.values() if v > 1)
        
        if new_cost < current_cost or random.random() < pow(2.718, -(new_cost - current_cost) / T):
            current_cost = new_cost
            if conflicts == 0: # Valid
                if max_val < best_cost:
                    best_cost = max_val
                    best_rows = [r[:] for r in rows]
        else:
            # Revert
            rows[r][j] = old_v
            current_diffs[r] = old_diffs
            
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    return best_rows
