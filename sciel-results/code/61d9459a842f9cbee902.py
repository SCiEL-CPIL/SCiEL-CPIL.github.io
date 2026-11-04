"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs(rows):
    """Return the set of all positive differences across all rows."""
    all_d = set()
    for r in rows:
        all_d |= _row_diffs(r)
    return all_d


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _local_search(rows, k, deadline):
    """Local search: try to reduce the max entry by moving elements to smaller positions."""
    best = [list(r) for r in rows]
    best_scope = max(max(r) for r in best)
    
    while time.time() < deadline - 5.0:
        improved = False
        # Try to reduce the maximum entry across all rows
        current_scope = max(max(r) for r in best)
        
        # Find the row with the maximum entry
        max_row_idx = max(range(len(best)), key=lambda i: max(best[i]))
        
        # Try to move elements in that row to smaller positions
        row = best[max_row_idx]
        for pos in range(len(row) - 1, 0, -1):  # from last element backwards
            if time.time() > deadline - 5.0:
                break
            old_val = row[pos]
            # Try moving to a smaller value
            for new_val in range(row[pos - 1] + 1, old_val):
                if time.time() > deadline - 5.0:
                    break
                row[pos] = new_val
                # Check validity
                if _valid(best):
                    new_scope = max(max(r) for r in best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        improved = True
                        break
                row[pos] = old_val
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try a different strategy: swap elements between rows
            found_swap = False
            for i in range(len(best)):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i + 1, len(best)):
                    if time.time() > deadline - 5.0:
                        break
                    # Try swapping the last elements of rows i and j
                    if len(best[i]) == len(best[j]):
                        saved_i = best[i][-1]
                        saved_j = best[j][-1]
                        if saved_i != saved_j:
                            best[i][-1] = saved_j
                            best[j][-1] = saved_i
                            if _valid(best):
                                new_scope = max(max(r) for r in best)
                                if new_scope < best_scope:
                                    best_scope = new_scope
                                    found_swap = True
                                    break
                            best[i][-1] = saved_i
                            best[j][-1] = saved_j
                    if found_swap:
                        break
                if found_swap:
                    break
            
            if not found_swap:
                # No improvement possible with simple moves, try random restarts
                if random.random() < 0.3:
                    # Perturb: randomly swap two elements in the same row
                    ridx = random.randint(0, len(best) - 1)
                    if len(best[ridx]) >= 3:
                        p1 = random.randint(1, len(best[ridx]) - 2)
                        p2 = random.randint(p1 + 1, len(best[ridx]) - 1)
                        best[ridx][p1], best[ridx][p2] = best[ridx][p2], best[ridx][p1]
                        if not _valid(best):
                            best[ridx][p1], best[ridx][p2] = best[ridx][p2], best[ridx][p1]
                            continue
                else:
                    break
    
    return best, best_scope


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
    
    # Local search to improve
    if time.time() < deadline - 10.0:
        local_rows, local_scope = _local_search(best_rows, k, deadline)
        if local_scope < best_scope:
            best_rows = local_rows
            best_scope = local_scope
    
    # Try to improve with backtracking on the last few rows
    # Strategy: fix the first (n_rows - 2) rows from greedy, then search the last 2
    if n_rows >= 2 and time.time() < deadline - 15.0:
        fixed_rows = best_rows[:n_rows - 2]
        fixed_used = set()
        for r in fixed_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    fixed_used.add(r[j] - r[i])
        
        # Search for better last 2 rows
        max_search_scope = best_scope
        for target_scope in range(max_search_scope - 1, max(best_scope - 20, 0), -1):
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
    
    return best_rows


def _search_two_rows(fixed_used, k, max_val, deadline):
    """Search for two rows of length k+1 starting at 0, max entry <= max_val,
    with all pairwise differences distinct from fixed_used and each other."""
    if time.time() > deadline:
        return None
    
    # Precompute valid single rows
    valid_rows = []
    for r1 in _enumerate_rows(k, max_val, fixed_used, deadline):
        used1 = _row_diffs(r1)
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


def _enumerate_rows(k, max_val, used, deadline):
    """Enumerate all valid rows of length k+1 starting at 0 with max <= max_val."""
    rows = []
    row = [0]
    row_diffs = set()
    
    def backtrack(pos):
        if time.time() > deadline:
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
    
    backtrack(1)
    return rows
# EVOLVE-BLOCK-END
