import time
import random
from collections import Counter

def construct(n_rows, k):
    deadline = time.time() + 145.0
    
    # Helper functions
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def max_entry(rows):
        return max(max(row) for row in rows)

    # Phase 1: Greedy Initialization
    def greedy_init():
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
        return rows

    best_rows = greedy_init()
    best_max = max_entry(best_rows)
    
    # Phase 2: DFS Search for better solution
    # We try to find a solution with max < best_max
    # We fix the first row to be the one from greedy? Or try all?
    # Let's try to construct rows one by one.
    
    used_diffs = set()
    current_rows = []
    
    def dfs(row_idx, pos_idx, curr_row):
        nonlocal best_rows, best_max, used_diffs, current_rows, deadline
        
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            # Found a complete solution
            current_max = max(max(r) for r in current_rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
            return True # Found a valid solution (maybe not the best, but valid)
            
        if pos_idx == k + 1:
            # Current row is complete
            # Check if we can prune: if the last element is already >= best_max, no need to continue?
            # Actually, if curr_row[-1] >= best_max, it won't improve best_max, so we can prune.
            if curr_row[-1] >= best_max:
                return False
                
            # Add row to current_rows
            current_rows.append(curr_row)
            # Note: used_diffs is already updated as we built the row?
            # Yes, we update used_diffs when adding elements to curr_row.
            
            success = dfs(row_idx + 1, 1, [0])
            
            # Backtrack
            current_rows.pop()
            return success
            
        # Determine range for curr_row[pos_idx]
        lower = curr_row[pos_idx - 1] + 1
        # Upper bound: we want to improve best_max, so we can cap at best_max - 1?
        # But we also need to fit the remaining elements.
        # A simple upper bound is best_max - (k - pos_idx)
        upper = best_max - (k - pos_idx)
        
        # If lower > upper, no solution
        if lower > upper:
            return False
            
        # Try values from lower to upper
        # Randomize order? No, smaller is better for scope.
        # But maybe randomization helps escape dead ends?
        # Let's stick to increasing order first.
        
        for val in range(lower, upper + 1):
            # Check conflicts with current row
            new_diffs = set()
            conflict = False
            for prev_val in curr_row:
                d = val - prev_val
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
                
            if conflict:
                continue
                
            # Check if new_diffs have internal conflicts?
            # No, differences within a row are distinct if the row is strictly increasing?
            # Wait. If row is [0, 2, 5], diffs are 2, 5, 3. Distinct.
            # If row is [0, 1, 3], diffs are 1, 3, 2. Distinct.
            # Is it always distinct for strictly increasing integers?
            # No. [0, 1, 2] -> 1, 2, 1. Conflict!
            # So we must check if new_diffs are distinct.
            if len(new_diffs) != len(curr_row):
                continue
                
            # Also check if any new_diff is already in used_diffs
            # We checked `d in used_diffs` for each d.
            # But we also need to ensure the new diffs don't conflict with each other?
            # The set `new_diffs` handles uniqueness among themselves.
            # The check `d in used_diffs` handles conflicts with previous rows.
            
            # Update state
            curr_row.append(val)
            used_diffs.update(new_diffs)
            
            # Recurse
            if dfs(row_idx, pos_idx + 1, curr_row):
                # Backtrack
                curr_row.pop()
                used_diffs.difference_update(new_diffs)
                return True # Found a solution, stop searching?
                # We want the BEST solution. So we should continue searching?
                # But DFS finds the first solution in the tree. If we order values ascending,
                # the first solution found will have the smallest possible last element for this branch.
                # However, a different branch (different earlier choices) might yield a smaller global max.
                # So we should NOT return True immediately if we just found a valid solution.
                # We should update best and continue?
                # But the search space is huge.
                # Let's return True to stop the current branch, but let the caller decide whether to continue.
                # Actually, if we find a solution with max < best_max, we update best_max.
                # Then the `upper` bound in the parent calls shrinks, pruning more.
                # So we should continue the search to see if we can find an even smaller one.
                # But returning True usually stops the DFS.
                # Let's modify: return False to continue searching, but update best.
                # Wait, if we return False, the parent will try the next value.
                # This is correct for finding the global minimum.
            
            # Backtrack
            curr_row.pop()
            used_diffs.difference_update(new_diffs)
            
        return False

    # Start DFS
    # We can try to fix the first row to be the greedy one to anchor the search?
    # Or just let it search freely.
    # Let's let it search freely, but we can bias it.
    
    # To make it faster, we can use a randomized restart approach for DFS.
    # But first, let's just run it once.
    
    # Note: The DFS above updates `best_max` inside.
    # We need to be careful with `used_diffs` initialization.
    
    # Let's run the DFS
    # We need to reset used_diffs and current_rows for each top-level call?
    # No, the recursion handles it.
    
    # We'll run the DFS for a certain time.
    dfs(0, 1, [0])
    
    # Phase 3: Simulated Annealing to polish
    # Start from best_rows
    current_rows = [r[:] for r in best_rows]
    
    # Precompute diff counts
    diff_counts = Counter()
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diff_counts[row[j] - row[i]] += 1
                
    def get_violations():
        return sum(c - 1 for c in diff_counts.values() if c > 1)
        
    current_violations = get_violations()
    current_max = max_entry(current_rows)
    current_cost = current_violations * 10000 + current_max
    
    T = 50.0
    T_min = 0.01
    cooling = 0.99995
    
    indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        r, i = random.choice(indices)
        old_val = current_rows[r][i]
        
        lower = current_rows[r][i - 1] + 1
        upper = current_rows[r][i + 1] - 1 if i < k else current_rows[r][k] + 50
        
        if random.random() < 0.3:
            span = upper - lower + 1
            new_val = lower + int(random.random() * span * 0.5)
        else:
            new_val = random.randint(lower, upper)
            
        if new_val == old_val:
            continue
            
        if new_val <= lower or (i < k and new_val >= upper):
            continue
            
        row = current_rows[r]
        removed_diffs = []
        added_diffs = []
        
        for j in range(len(row)):
            if j == i:
                continue
            if j < i:
                old_d = old_val - row[j]
                new_d = new_val - row[j]
                removed_diffs.append(old_d)
                added_diffs.append(new_d)
            else:
                old_d = row[j] - old_val
                new_d = row[j] - new_val
                removed_diffs.append(old_d)
                added_diffs.append(new_d)
                
        for d in removed_diffs:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
        for d in added_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
        new_violations = get_violations()
        new_max = max(max(row) for row in current_rows)
        new_cost = new_violations * 10000 + new_max
        
        delta = new_cost - current_cost
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            current_rows[r][i] = new_val
            current_violations = new_violations
            current_max = new_max
            current_cost = new_cost
            
            if new_max < best_max:
                best_max = new_max
                best_rows = [rr[:] for rr in current_rows]
        else:
            for d in added_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in removed_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            current_rows[r][i] = old_val
            
        T *= cooling
        
    return best_rows
