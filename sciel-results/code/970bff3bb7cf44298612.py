"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110
    
    # Get initial greedy solution
    rows = _greedy_construct(n_rows, k)
    best_rows = [row[:] for row in rows]
    best_max = max(max(row) for row in best_rows)
    
    # Local search: try to reduce max by swapping elements
    best_rows = _local_search(best_rows, n_rows, k, start_time, deadline)
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Standard greedy: each row takes smallest available marks."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _get_row_diffs(row):
    """Get all pairwise differences for a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _validate_full(rows):
    """Check all differences are distinct across all rows."""
    all_diffs = []
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                all_diffs.append(row[j] - row[i])
    return len(all_diffs) == len(set(all_diffs))


def _local_search(rows, n_rows, k, start_time, deadline):
    """Local search to reduce maximum element."""
    best_rows = [row[:] for row in rows]
    best_max = max(max(row) for row in best_rows)
    no_improve = 0
    
    while time.time() < deadline - 5 and no_improve < 30:
        improved = False
        
        # Compute difference sets
        row_diffs_list = [_get_row_diffs(row) for row in best_rows]
        all_diffs = set()
        for rd in row_diffs_list:
            all_diffs |= rd
        
        # Try to improve the row with the largest element first
        max_idx = max(range(n_rows), key=lambda i: max(best_rows[i]))
        order = [max_idx] + [i for i in range(n_rows) if i != max_idx]
        
        for r_idx in order:
            if time.time() > deadline - 5:
                break
            
            row = best_rows[r_idx]
            other_diffs = all_diffs - row_diffs_list[r_idx]
            current_max = max(row)
            
            # Try replacing individual elements with smaller values
            # Focus on the largest element first
            positions = list(range(len(row) - 1, 0, -1))  # from end to start
            
            for pos in positions:
                if time.time() > deadline - 5:
                    break
                
                old_val = row[pos]
                if old_val <= 1:
                    continue
                
                # Try replacing with smaller values
                # Must maintain: row[pos-1] < new_val < row[pos+1] (if exists)
                lower = row[pos - 1] + 1
                upper = row[pos + 1] - 1 if pos + 1 < len(row) else old_val - 1
                
                for new_val in range(old_val - 1, lower - 1, -1):
                    if time.time() > deadline - 5:
                        break
                    
                    # Check if new_val creates conflicts
                    # Differences involving new_val:
                    # new_val - row[i] for i < pos
                    # row[j] - new_val for j > pos
                    conflict = False
                    for i in range(pos):
                        d = new_val - row[i]
                        if d in other_diffs:
                            conflict = True
                            break
                        # Also check if d is already in this row's diffs (shouldn't be if new_val != old_val)
                    if conflict:
                        continue
                    
                    for j in range(pos + 1, len(row)):
                        d = row[j] - new_val
                        if d in other_diffs:
                            conflict = True
                            break
                    if conflict:
                        continue
                    
                    # Also need to check that the new differences don't conflict
                    # with existing differences in this row
                    # The differences not involving pos remain the same
                    # New differences: new_val - row[i] for i < pos, and row[j] - new_val for j > pos
                    # These must not be in other_diffs (checked above)
                    # And they must be distinct from each other (they are, since new_val is fixed)
                    # And they must be distinct from old differences in this row that don't involve pos
                    # Old diffs not involving pos: row[j] - row[i] for i,j != pos, i < j
                    # These are unchanged, so no conflict there
                    
                    # But we also need: new_val - row[i] != row[j] - new_val for i < pos, j > pos
                    # i.e., 2*new_val != row[i] + row[j]
                    # And new_val - row[i] != row[j] - row[i'] for i < i' < pos, j > pos
                    # etc. This is getting complex. Let me just validate the full row.
                    
                    test_row = row[:]
                    test_row[pos] = new_val
                    test_row.sort()  # Ensure sorted
                    
                    # Check if test_row is strictly increasing and starts at 0
                    if test_row[0] != 0:
                        continue
                    for t in range(len(test_row) - 1):
                        if test_row[t + 1] <= test_row[t]:
                            continue
                    
                    # Check all differences in test_row
                    test_diffs = _get_row_diffs(test_row)
                    # Check no conflict with other rows
                    if test_diffs & other_diffs:
                        continue
                    # Check internal consistency (all diffs distinct within row)
                    # This is guaranteed if row is strictly increasing and we computed diffs correctly
                    # But let's verify count
                    expected_count = len(test_row) * (len(test_row) - 1) // 2
                    if len(test_diffs) != expected_count:
                        continue
                    
                    # Full validation
                    test_rows = [r[:] for r in best_rows]
                    test_rows[r_idx] = test_row
                    if _validate_full(test_rows):
                        best_rows[r_idx] = test_row
                        best_max = max(max(r) for r in best_rows)
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    
    return best_rows

# EVOLVE-BLOCK-END
