"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin under 150s wall-clock

    # Each row has k+1 entries; index 0 is fixed at 0.
    m = k + 1  # number of entries per row
    # Total possible distinct differences if max entry is M: M*(M+1)/2
    # We'll track used differences in a set.

    def compute_row_diffs(row):
        """Return set of positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d > 0:
                    diffs.add(d)
        return diffs

    def compute_all_diffs(rows):
        """Return (used_set, conflict_count, max_entry)."""
        used = set()
        conflicts = 0
        max_entry = 0
        for row in rows:
            for d in compute_row_diffs(row):
                if d in used:
                    conflicts += 1
                else:
                    used.add(d)
            if row[-1] > max_entry:
                max_entry = row[-1]
        return used, conflicts, max_entry

    def row_conflicts(row, used_without_row):
        """Return set of differences in row that conflict with used_without_row, plus internal duplicates count."""
        diffs = compute_row_diffs(row)
        conflict_diffs = diffs & used_without_row
        # Internal duplicates are already handled by set, but we need count
        # Actually, if a diff appears twice in the same row, it's one entry in set.
        # But the problem says "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows"
        # So within a row, differences must also be distinct? Let me re-read.
        # "all differences between two entries of the same row are distinct over all rows"
        # This means: for any two pairs (from possibly different rows), the differences are distinct.
        # So within a single row, differences must be distinct too.
        # A set naturally handles uniqueness within the row.
        return conflict_diffs

    def repair_row(row, used_global, row_idx, max_iter=50):
        """Try to fix conflicts in row by adjusting entries. Returns (new_row, success)."""
        best_row = list(row)
        best_conflicts = len(compute_row_diffs(best_row) & used_global)
        
        for _ in range(max_iter):
            # Pick a random position to modify (not the first which is 0)
            pos = random.randint(1, len(best_row) - 1)
            old_val = best_row[pos]
            # Try a random new value
            new_val = old_val + random.randint(-5, 5)
            if new_val <= 0 or new_val >= best_row[pos + 1] if pos + 1 < len(best_row) else new_val <= 0:
                continue
            if pos + 1 < len(best_row) and new_val >= best_row[pos + 1]:
                continue
            if pos > 0 and new_val <= best_row[pos - 1]:
                continue
            
            test_row = list(best_row)
            test_row[pos] = new_val
            test_diffs = compute_row_diffs(test_row)
            test_conflicts = len(test_diffs & used_global)
            
            if test_conflicts < best_conflicts:
                best_row = test_row
                best_conflicts = test_conflicts
                if best_conflicts == 0:
                    break
        
        return best_row, best_conflicts < len(compute_row_diffs(row) & used_global) or best_conflicts == 0

    # Initialize: use greedy to get a starting point
    used_init, rows_init = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < m:
            x += 1
            new = {x - a for a in row}
            if not new & used_init:
                row.append(x)
                used_init |= new
        rows_init.append(row)
    
    best_rows = [list(r) for r in rows_init]
    best_used, best_conflicts, best_max = compute_all_diffs(best_rows)
    best_score = best_conflicts * 1000 + best_max  # combined objective
    
    # Simulated annealing
    temp = 10.0
    cooling = 0.995
    min_temp = 0.1
    
    while temp > min_temp:
        if time.time() > deadline:
            break
        
        # Pick a random row to modify
        row_idx = random.randint(0, n_rows - 1)
        old_row = list(best_rows[row_idx])
        
        # Generate a perturbed version of the row
        new_row = list(old_row)
        # Modify 1-3 random positions
        n_mods = random.randint(1, 3)
        for _ in range(n_mods):
            pos = random.randint(1, len(new_row) - 1)
            delta = random.randint(-3, 3)
            if delta == 0:
                continue
            new_val = new_row[pos] + delta
            # Check bounds
            if new_val <= 0:
                continue
            if pos > 0 and new_val <= new_row[pos - 1]:
                continue
            if pos + 1 < len(new_row) and new_val >= new_row[pos + 1]:
                continue
            new_row[pos] = new_val
        
        if new_row == old_row:
            continue
        
        # Compute new state
        test_rows = [list(r) if i != row_idx else new_row for i, r in enumerate(best_rows)]
        test_used, test_conflicts, test_max = compute_all_diffs(test_rows)
        test_score = test_conflicts * 1000 + test_max
        
        delta_score = test_score - best_score
        
        if delta_score <= 0 or random.random() < pow(2.718281828, -delta_score / temp):
            best_rows = test_rows
            best_used, best_conflicts, best_max = test_used, test_conflicts, test_max
            best_score = test_score
        
        temp *= cooling
    
    # Final repair pass: try to fix any remaining conflicts
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Find rows with conflicts
        used_set = set()
        row_diffs_list = []
        for i, row in enumerate(best_rows):
            diffs = compute_row_diffs(row)
            row_diffs_list.append(diffs)
            used_set |= diffs
        
        # Find conflicts: diffs that appear in multiple rows
        diff_count = {}
        for diffs in row_diffs_list:
            for d in diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
        
        conflicting_diffs = {d for d, c in diff_count.items() if c > 1}
        
        if not conflicting_diffs:
            break
        
        # Pick a random conflicting diff and try to fix one of the rows containing it
        target_diff = random.choice(list(conflicting_diffs))
        involved_rows = [i for i, diffs in enumerate(row_diffs_list) if target_diff in diffs]
        
        if len(involved_rows) < 2:
            continue
        
        row_idx = random.choice(involved_rows)
        # Try to modify this row to avoid target_diff
        new_row = list(best_rows[row_idx])
        found_fix = False
        
        for _ in range(20):
            pos = random.randint(1, len(new_row) - 1)
            delta = random.randint(-5, 5)
            if delta == 0:
                continue
            new_val = new_row[pos] + delta
            if new_val <= 0:
                continue
            if pos > 0 and new_val <= new_row[pos - 1]:
                continue
            if pos + 1 < len(new_row) and new_val >= new_row[pos + 1]:
                continue
            
            test_row = list(new_row)
            test_row[pos] = new_val
            test_diffs = compute_row_diffs(test_row)
            
            # Check if target_diff is gone and no new conflicts
            if target_diff not in test_diffs:
                # Check against all other rows' diffs
                other_diffs = set()
                for i, row in enumerate(best_rows):
                    if i != row_idx:
                        other_diffs |= compute_row_diffs(row)
                
                new_conflicts = test_diffs & other_diffs
                old_conflicts = row_diffs_list[row_idx] & other_diffs
                
                if len(new_conflicts) <= len(old_conflicts):
                    best_rows[row_idx] = test_row
                    found_fix = True
                    break
        
        if not found_fix:
            continue
    
    # Ensure validity: if still conflicts, fall back to greedy
    final_used, final_conflicts, final_max = compute_all_diffs(best_rows)
    if final_conflicts > 0:
        # Fallback to greedy
        used_greedy, rows_greedy = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < m:
                x += 1
                new = {x - a for a in row}
                if not new & used_greedy:
                    row.append(x)
                    used_greedy |= new
            rows_greedy.append(row)
        return rows_greedy
    
    return best_rows
# EVOLVE-BLOCK-END
