"""
Difference triangle set (n_rows, k) solver.

Strategy:
1. Compute a lower bound (LB) on the scope (largest entry) using counting / packing.
2. Use a randomized backtracking search with forward checking to find a valid set
   whose largest entry is as small as possible.
3. Iterate over candidate scopes M from LB upwards, using a time budget to stop
   early and return the best solution found.
"""

import time
import random
from itertools import combinations


def _greedy_row(used, k):
    """Build one row greedily: choose the smallest possible next value."""
    row = [0]
    x = 1
    while len(row) < k + 1:
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
        x += 1
    return row


def _greedy_full(n_rows, k):
    """Build all rows greedily (baseline)."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
            x += 1
        rows.append(row)
        used |= {b - a for a, b in combinations(row, 2)}
    return rows


def _lower_bound(n_rows, k):
    """
    Lower bound on the largest entry M.
    Total number of distinct differences needed: n_rows * k*(k+1)/2
    Each difference d in 1..M can appear at most once.
    So we need M >= n_rows * k*(k+1)/2  ... but that's a very loose bound.
    
    Better: For a single row with k+1 elements, the differences are at most
    k*(k+1)/2 distinct values, each <= M.
    Across n_rows rows, we need n_rows * k*(k+1)/2 distinct differences in [1, M].
    So M >= n_rows * k*(k+1)/2.
    
    For n_rows=7, k=5: 7*15 = 105. This matches the stated counting bound.
    """
    return n_rows * k * (k + 1) // 2


def _is_valid(rows, k):
    """Check that all rows are valid and all differences are distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _max_entry(rows):
    return max(max(row) for row in rows) if rows else 0


def _backtrack(n_rows, k, M, deadline, rng):
    """
    Try to find n_rows rows, each with k+1 elements in [0, M], increasing,
    starting at 0, such that all pairwise differences are distinct across rows.
    """
    used = set()
    rows = []

    def search(row_idx, pos, current_row):
        """
        row_idx: which row we're building (0-indexed)
        pos: next position in current_row (0-indexed, so current_row has `pos` elements)
        current_row: list of elements so far
        """
        if time.monotonic() > deadline:
            return None

        if pos == k + 1:
            # Row complete
            new_rows = rows + [current_row]
            if row_idx + 1 == n_rows:
                return new_rows
            # Recurse to next row
            result = search(row_idx + 1, 0, [0])
            return result

        # Determine candidate range for next element
        if pos == 0:
            # First element must be 0
            candidates = [0]
        else:
            start = current_row[-1] + 1
            end = M
            candidates = list(range(start, end + 1))

        # Shuffle candidates for randomization
        rng.shuffle(candidates)

        # Forward checking: for each candidate, check if it creates a conflict
        # and also whether we can still fill the remaining positions
        for val in candidates:
            # Compute new differences
            new_diffs = {val - a for a in current_row}
            if new_diffs & used:
                continue

            # Check if remaining positions can be filled:
            # We need (k + 1 - pos - 1) more elements, all > val, <= M
            # Minimum next value would be val + 1, so we need:
            # val + (k + 1 - pos - 1) <= M
            remaining = k + 1 - pos - 1
            if val + remaining > M:
                continue

            # Also, the new differences must not conflict with each other
            # (they are all distinct by construction since current_row is increasing)
            
            # Recurse
            result = search(row_idx, pos + 1, current_row + [val])
            if result is not None:
                return result

        return None

    return search(0, 0, [0])


def _search_with_budget(n_rows, k, M, time_budget, rng):
    """
    Try to find a solution with max entry <= M using randomized backtracking.
    Multiple attempts with different random seeds.
    """
    start = time.monotonic()
    attempts = 0
    max_attempts = 5000  # cap to avoid infinite loop if very easy
    
    while time.monotonic() - start < time_budget and attempts < max_attempts:
        result = _backtrack(n_rows, k, M, time.monotonic() + time_budget, rng)
        if result is not None:
            return result
        attempts += 1
        # Small pause to let time check work
        if attempts % 100 == 0:
            time.sleep(0.001)
    
    return None


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct over all rows.
    Minimize the largest entry.
    """
    start_time = time.monotonic()
    total_deadline = start_time + 140.0  # leave 10s margin for 150s limit
    
    rng = random.Random(42)
    
    # First, get a baseline greedy solution
    greedy_rows = _greedy_full(n_rows, k)
    best_rows = greedy_rows
    best_scope = _max_entry(greedy_rows)
    
    # Lower bound
    lb = _lower_bound(n_rows, k)
    
    # Search for better solutions
    # Start from lb, go up until we find a solution or run out of time
    # Allocate time per M value
    # We'll try a range of M values
    
    # Estimate: how many M values can we try?
    # Give each M value a fraction of the remaining time
    # Start with lb, then lb+1, etc.
    
    # Actually, let's be smarter: 
    # 1. Try M = lb first with a generous time budget
    # 2. If that fails, try M = lb+1, etc.
    # 3. Keep track of the best solution found
    
    # Time allocation: start with 30s for the first M, then decrease
    time_remaining = total_deadline - time.monotonic()
    
    # Try M values from lb upwards
    m = lb
    max_m_tried = lb + 20  # don't go too far above LB
    
    while m <= max_m_tried and time.monotonic() < total_deadline:
        time_left = total_deadline - time.monotonic()
        if time_left < 5.0:
            break
        
        # Allocate a portion of remaining time for this M
        # Earlier M values get more time
        fraction = 1.0 / (m - lb + 1)
        budget = min(time_left * fraction, 30.0)
        budget = max(budget, 2.0)  # minimum 2 seconds
        
        result = _search_with_budget(n_rows, k, m, budget, rng)
        if result is not None:
            scope = _max_entry(result)
            if scope < best_scope:
                best_rows = result
                best_scope = scope
            # Found a solution at this M, try to improve
            # But also try slightly smaller? No, m starts at lb.
            # Since m >= lb and we found a solution, this is good.
            # We can break if m == lb (optimal) or continue to see if we can find
            # a solution with smaller scope... but m is already at lb.
            # Actually, the solution might have max_entry < m.
            # So we should check if we can try m-1? No, m starts at lb.
            # If we found a solution, best_scope <= m.
            # We can stop if best_scope == lb.
            if best_scope == lb:
                break
            # Continue to see if a different random seed finds a better one
            # at the same M, or try M+1 in case the search was too limited
            # Actually, if we found a solution at M, the scope is <= M.
            # We should try to minimize further. Since M is increasing, and we
            # start at LB, the first M that yields a solution gives scope <= M.
            # We can stop and return, or try a bit more.
            # Let's try a few more random seeds at the same M to see if we
            # can get a smaller scope (unlikely since scope <= M, but possible
            # that the solution doesn't use the full M).
            # Actually, the scope is the max entry, which is <= M.
            # We can't get below lb. So if best_scope == lb, we're done.
            # Otherwise, we can try more seeds at this M, or move to M+1.
            # Moving to M+1 won't help since scope is already <= M.
            # So let's just do more attempts at this M.
            continue
        
        # No solution found at this M, try next M
        m += 1
    
    # Final check: ensure best_rows is valid
    if not _is_valid(best_rows, k):
        # Fallback to greedy
        best_rows = _greedy_full(n_rows, k)
    
    return best_rows
