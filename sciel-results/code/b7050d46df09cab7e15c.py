"""Difference triangle set (7,5): minimize the largest entry.
Strategy: branch-and-bound on rows with strong pruning, plus targeted local improvement.
"""
import time
import random
import sys

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
    """Return the union of all positive differences across all rows."""
    all_d = set()
    for row in rows:
        all_d |= _row_diffs(row)
    return all_d


def _check_valid(rows):
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


def _greedy_rows(n_rows, k, used=None):
    """Greedy construction of n_rows rows."""
    if used is None:
        used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows, used


def _can_complete(row, pos, k, used, max_allowed, deadline):
    """Quick feasibility check: can we extend row to length k+1 within max_allowed?
    Uses a greedy lower bound on the next value needed."""
    if time.time() > deadline:
        return False
    remaining = k + 1 - len(row)
    if remaining <= 0:
        return True
    # Lower bound: each new element adds at least 1 to the max
    # and needs (len(row)) new differences
    # Simple bound: max_allowed must be at least row[-1] + remaining
    if row[-1] + remaining > max_allowed:
        return False
    return True


def _search_row(row, pos, k, used, max_allowed, deadline, best_rows, best_scope):
    """Backtracking search for a row of length k+1 starting with current row.
    Returns the completed row if found, else None."""
    if time.time() > deadline:
        return None
    if pos == k + 1:
        return list(row)
    
    # Pruning: if even the best possible completion exceeds current best scope, skip
    if max_allowed >= best_scope:
        # We still might find a valid row that helps other rows, but for scope minimization
        # this row alone doesn't improve things. However, we need valid rows for the set.
        # Actually, we're searching for ANY valid row within max_allowed.
        pass
    
    # Try values in increasing order (greedy bias for smaller max)
    start = row[-1] + 1
    for x in range(start, max_allowed + 1):
        if time.time() > deadline:
            return None
        new_diffs = {x - a for a in row}
        if new_diffs & used:
            continue
        row.append(x)
        result = _search_row(row, pos + 1, k, used, max_allowed, deadline, best_rows, best_scope)
        if result is not None:
            return result
        row.pop()
    return None


def _try_replace_row(rows, row_idx, k, max_allowed, deadline):
    """Try to find a replacement for rows[row_idx] with max entry <= max_allowed.
    Returns the new row if found, else None."""
    other_rows = rows[:row_idx] + rows[row_idx + 1:]
    other_used = set()
    for r in other_rows:
        other_used |= _row_diffs(r)
    
    row = [0]
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == k + 1:
            return True
        for x in range(row[-1] + 1, max_allowed + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & other_used:
                continue
            row.append(x)
            if backtrack(pos + 1):
                return True
            row.pop()
        return False
    
    if backtrack(1):
        return list(row)
    return None


def _local_search(rows, k, deadline):
    """Iteratively try to replace each row with one that has a smaller max entry."""
    best = [list(r) for r in rows]
    best_scope = max(max(r) for r in best)
    no_improve_count = 0
    
    while time.time() < deadline - 3.0:
        improved = False
        
        # Try each row in order of largest max first
        order = sorted(range(len(best)), key=lambda i: -max(best[i]))
        
        for row_idx in order:
            if time.time() > deadline - 3.0:
                break
            current_max = max(best[row_idx])
            
            # Try to find a row with max < current_max
            for target_max in range(current_max - 1, max(current_max - 15, 0), -1):
                if time.time() > deadline - 3.0:
                    break
                new_row = _try_replace_row(best, row_idx, k, target_max, deadline)
                if new_row is not None:
                    best[row_idx] = new_row
                    new_scope = max(max(r) for r in best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        # Keep the new row anyway, it might help later
                        pass
                if improved:
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 4:
                # Try random perturbation: replace a random row
                ridx = random.randint(0, len(best) - 1)
                old_row = best[ridx]
                new_row = _try_replace_row(best, ridx, k, max(old_row), deadline)
                if new_row is not None and new_row != old_row:
                    best[ridx] = new_row
                    no_improve_count = 0
                else:
                    break
        else:
            no_improve_count = 0
    
    return best, best_scope


def _branch_and_bound(n_rows, k, deadline):
    """Branch and bound: build rows one at a time, trying to minimize the overall scope.
    Uses the fact that we can search for each row with a tight max bound."""
    best_rows = None
    best_scope = float('inf')
    
    # Start with greedy baseline
    greedy_rows, greedy_used = _greedy_rows(n_rows, k)
    best_rows = greedy_rows
    best_scope = max(max(r) for r in best_rows)
    
    # Now try to improve by searching for better rows
    # Strategy: for each row, try to find a replacement with smaller max
    # Use decreasing target max
    
    for iteration in range(20):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        # Try rows in random order to escape local minima
        indices = list(range(n_rows))
        random.shuffle(indices)
        
        for row_idx in indices:
            if time.time() > deadline - 5.0:
                break
            current_max = max(best_rows[row_idx])
            
            # Try to reduce this row's max
            for target_max in range(current_max - 1, max(current_max - 20, 0), -1):
                if time.time() > deadline - 5.0:
                    break
                new_row = _try_replace_row(best_rows, row_idx, k, target_max, deadline)
                if new_row is not None:
                    new_scope = max(max(r) for r in best_rows[:row_idx] + [new_row] + best_rows[row_idx+1:])
                    if new_scope < best_scope:
                        best_rows[row_idx] = new_row
                        best_scope = new_scope
                        improved = True
                        break
                    else:
                        # Accept if same scope but different row (might help later)
                        best_rows[row_idx] = new_row
                        break
        
        if not improved:
            # Try a more aggressive approach: rebuild from scratch with tighter bounds
            if time.time() > deadline - 10.0:
                break
            
            # Try building all rows with a global max bound
            for global_max in range(best_scope - 1, max(best_scope - 10, 0), -1):
                if time.time() > deadline - 10.0:
                    break
                
                # Try to construct n_rows rows all with max <= global_max
                used = set()
                rows = []
                success = True
                for r in range(n_rows):
                    row = [0]
                    def backtrack_row(pos, current_row, current_used):
                        if time.time() > deadline - 10.0:
                            return False
                        if pos == k + 1:
                            return True
                        for x in range(current_row[-1] + 1, global_max + 1):
                            new_diffs = {x - a for a in current_row}
                            if new_diffs & current_used:
                                continue
                            current_row.append(x)
                            if backtrack_row(pos + 1, current_row, current_used):
                                return True
                            current_row.pop()
                        return False
                    
                    if backtrack_row(1, row, used):
                        rows.append(row)
                        used |= _row_diffs(row)
                    else:
                        success = False
                        break
                
                if success:
                    scope = max(max(r) for r in rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = rows
                    break
    
    return best_rows, best_scope


def construct(n_rows, k):
    deadline = time.time() + 145.0  # safety margin before 150s kill
    best_rows = None
    best_scope = float('inf')
    
    # Greedy baseline
    greedy_rows, greedy_used = _greedy_rows(n_rows, k)
    best_rows = greedy_rows
    best_scope = max(max(r) for r in best_rows)
    
    # Branch and bound improvement
    if time.time() < deadline - 10.0:
        bb_rows, bb_scope = _branch_and_bound(n_rows, k, deadline)
        if bb_scope < best_scope:
            best_rows = bb_rows
            best_scope = bb_scope
    
    # Final local search pass
    if time.time() < deadline - 3.0:
        local_rows, local_scope = _local_search(best_rows, k, deadline)
        if local_scope < best_scope:
            best_rows = local_rows
            best_scope = local_scope
    
    return best_rows
# EVOLVE-BLOCK-END
