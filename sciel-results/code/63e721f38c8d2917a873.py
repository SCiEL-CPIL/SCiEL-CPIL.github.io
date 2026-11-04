"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 110.0  # safety margin before 120s kill
    best_rows = None
    best_scope = float('inf')
    
    # First do a quick greedy to get a valid baseline
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    best_rows = rows
    best_scope = max(max(r) for r in rows)
    
    # Try to improve with backtracking on the last few rows
    # Strategy: fix the first (n_rows - 2) rows from greedy, then search the last 2
    if n_rows >= 2:
        fixed_rows = rows[:n_rows - 2]
        fixed_used = set()
        for r in fixed_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    fixed_used.add(r[j] - r[i])
        
        # Search for better last 2 rows
        max_search_scope = best_scope
        for target_scope in range(max_search_scope - 1, best_scope - 20, -1):
            if time.time() > deadline - 5.0:
                break
            found = _search_two_rows(fixed_used, k, target_scope, deadline)
            if found is not None:
                candidate = fixed_rows + found
                scope = max(max(r) for r in candidate)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = candidate
                    break
    
    # Also try improving individual rows from the end
    if time.time() < deadline - 10.0:
        for row_idx in range(n_rows - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            # Try to find a better row at this position
            other_rows = best_rows[:row_idx] + best_rows[row_idx + 1:]
            other_used = set()
            for r in other_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        other_used.add(r[j] - r[i])
            
            current_row = best_rows[row_idx]
            current_max = max(current_row)
            
            # Try to find a row with smaller max
            for target_max in range(current_max - 1, max(current_max - 15, 0), -1):
                if time.time() > deadline - 5.0:
                    break
                new_row = _search_single_row(other_used, k, target_max, deadline)
                if new_row is not None:
                    candidate = best_rows[:row_idx] + [new_row] + best_rows[row_idx + 1:]
                    scope = max(max(r) for r in candidate)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = candidate
                    break
            # Update best_rows for subsequent iterations
            best_rows = best_rows if best_rows is not None else rows
    
    return best_rows


def _search_two_rows(fixed_used, k, max_val, deadline):
    """Search for two rows of length k+1 starting at 0, max entry <= max_val,
    with all pairwise differences distinct from fixed_used and each other."""
    if time.time() > deadline:
        return None
    
    # Precompute valid single rows
    valid_rows = []
    for r1 in _enumerate_rows(k, max_val, fixed_used):
        used1 = set()
        for i in range(len(r1)):
            for j in range(i + 1, len(r1)):
                used1.add(r1[j] - r1[i])
        valid_rows.append((r1, used1))
    
    # Find two disjoint rows
    for i in range(len(valid_rows)):
        if time.time() > deadline:
            return None
        r1, used1 = valid_rows[i]
        for j in range(i + 1, len(valid_rows)):
            if time.time() > deadline:
                return None
            r2, used2 = valid_rows[j]
            if not (used1 & used2):
                return [r1, r2]
    return None


def _search_single_row(used, k, max_val, deadline):
    """Search for a single row of length k+1 starting at 0, max entry <= max_val,
    with all differences disjoint from used."""
    if time.time() > deadline:
        return None
    
    row = [0]
    row_diffs = set()
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == k + 1:
            return True
        for x in range(row[-1] + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            if new_diffs & row_diffs:
                continue
            
            row.append(x)
            row_diffs.update(new_diffs)
            if backtrack(pos + 1):
                return True
            row_diffs.difference_update(new_diffs)
            row.pop()
        return False
    
    if backtrack(1):
        return list(row)
    return None


def _enumerate_rows(k, max_val, used):
    """Enumerate all valid rows of length k+1 starting at 0 with max <= max_val."""
    rows = []
    row = [0]
    row_diffs = set()
    
    def backtrack(pos):
        if time.time() > deadline_ref:
            return
        if pos == k + 1:
            rows.append(list(row))
            return
        for x in range(row[-1] + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            if new_diffs & row_diffs:
                continue
            
            row.append(x)
            row_diffs.update(new_diffs)
            backtrack(pos + 1)
            row_diffs.difference_update(new_diffs)
            row.pop()
    
    global deadline_ref
    deadline_ref = time.time() + 5.0
    backtrack(1)
    return rows
# EVOLVE-BLOCK-END
