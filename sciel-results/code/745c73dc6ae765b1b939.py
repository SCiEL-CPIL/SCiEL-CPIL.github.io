"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    # Deadline: 140 seconds (10s safety margin from 150s limit)
    deadline = time.time() + 140.0
    
    # 1. Generate a baseline greedy solution to have a valid fallback
    baseline = _greedy_construct(n_rows, k)
    best_rows = baseline
    best_score = _max_entry(baseline)
    
    if n_rows <= 2 and k <= 3:
        return baseline

    # Strategy: Multi-start Constraint Satisfaction + Conflict-Directed Local Search
    # 1. Try to build a valid solution from scratch using backtracking with forward checking.
    #    This is much faster than SA for finding *any* valid solution, especially if we
    #    order rows by "difficulty" (tightest constraints first) or use randomized restarts.
    # 2. Once we have a valid solution, try to improve it (reduce max entry) by:
    #    a. Reordering rows (to allow backtracking to find tighter solutions)
    #    b. Replacing specific rows with tighter ones (conflict-directed)
    #    c. Local search on individual elements (as before, but smarter)

    # --- Phase 1: Find a valid solution via Backtracking ---
    # We will try multiple times to construct a valid set of rows.
    # Each attempt uses randomized backtracking.
    
    best_valid_found = None
    best_valid_score = float('inf')
    
    # Try up to 50 backtracking attempts or until time runs out
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
            
        # Randomize the order of rows to break symmetry and explore different spaces
        # Actually, we construct rows sequentially. The order in which we *try* to fit rows
        # matters. Let's just generate rows one by one, but with randomization in the choice.
        
        rows, success = _backtrack_construct(n_rows, k, deadline - 10)
        
        if success:
            score = _max_entry(rows)
            if score < best_valid_score:
                best_valid_score = score
                best_valid_found = [row[:] for row in rows]
                
                # If we found a solution better than baseline, update global best
                if score < best_score:
                    best_rows = best_valid_found
                    best_score = score
                    
                # If we found a very good solution, we might stop early?
                # The counting bound is 105. If we hit 105, we can't do better (for this n,k).
                # But we don't know the exact bound for all n,k. Let's keep going to find lower.
                if score <= 105: 
                    break

    # If backtracking found a valid solution, use it as the starting point for refinement
    # Otherwise, use the baseline (which is always valid)
    current_rows = best_valid_found if best_valid_found else baseline
    current_score = _max_entry(current_rows)
    
    # --- Phase 2: Refine the valid solution ---
    # Goal: Reduce the maximum entry.
    # Method: 
    # 1. Identify the row with the largest entry.
    # 2. Try to replace it with a "tighter" row that fits in the remaining difference space.
    # 3. Try to shift elements within rows to reduce the max.
    
    if time.time() < deadline - 5:
        current_rows, current_score = _refine_solution(current_rows, n_rows, k, deadline)
        
        if current_score < best_score:
            best_rows = current_rows
            best_score = current_score

    # Final safety check
    if not _is_valid(best_rows, n_rows, k):
        return baseline
        
    return best_rows

def _backtrack_construct(n_rows, k, deadline):
    """
    Construct n_rows valid rows using backtracking with forward checking.
    Returns (rows, success).
    """
    used_diffs = set()
    rows = []
    
    def backtrack(idx):
        if idx == n_rows:
            return True
            
        if time.time() > deadline:
            return False
            
        # Generate candidates for the current row
        # We want rows that are "tight" (small max) but also likely to fit.
        # Strategy: Try to construct a row with max value bounded by a limit.
        # Start with a loose limit and tighten it? Or just random?
        # Let's try random generation first, but constrained by used_diffs.
        
        # To increase success rate, we can try to generate rows with smaller max values first.
        # But random is good for diversity.
        
        # Let's try a few random rows.
        for _ in range(20):  # 20 attempts per row index
            row = _generate_row_with_constraints(k, used_diffs, max_val=150)
            if row is None:
                continue
                
            row_diffs = _get_diffs(row)
            
            # Check if this row conflicts with already used diffs
            if row_diffs & used_diffs:
                continue
                
            # Accept this row
            rows.append(row)
            used_diffs.update(row_diffs)
            
            if backtrack(idx + 1):
                return True
            
            # Backtrack
            rows.pop()
            used_diffs -= row_diffs
            
        return False

    success = backtrack(0)
    return rows, success

def _generate_row_with_constraints(k, used_diffs, max_val=150):
    """
    Generate a single valid row of k+1 increasing integers starting at 0,
    such that its internal differences do not intersect with used_diffs.
    """
    # We build the row element by element.
    # At each step, we choose the next element such that:
    # 1. It is larger than the previous.
    # 2. The differences with all previous elements are not in used_diffs.
    # 3. The differences with all previous elements are distinct from each other (internal validity).
    
    row = [0]
    internal_diffs = set()
    
    for _ in range(k):
        last = row[-1]
        found = False
        # Try values starting from last+1 up to max_val
        # To make it "tight", we might prefer smaller values.
        # But randomization helps escape local minima.
        # Let's try a mix: 50% smallest available, 50% random.
        
        if random.random() < 0.5:
            # Smallest available
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                # Check internal distinctness
                if len(new_diffs) != len(row):
                    continue
                # Check against used_diffs
                if new_diffs & used_diffs:
                    continue
                # Check internal diffs against each other? 
                # new_diffs is a set, so if len == len(row), they are distinct.
                # But we also need to ensure they don't conflict with *existing* internal_diffs?
                # No, internal_diffs is just the set of diffs within the current row.
                # If we add x, the new diffs are {x - a for a in row}.
                # These must be distinct from each other (guaranteed by set size if we check carefully? No, x-a1 could equal x-a2 only if a1=a2, which is impossible).
                # Wait, x-a1 and x-a2 are distinct if a1 != a2.
                # So the new diffs are always distinct among themselves.
                # We just need to ensure they don't overlap with *previous* internal diffs?
                # Yes, all diffs in the row must be distinct.
                # So new_diffs must be disjoint from internal_diffs.
                
                if new_diffs & internal_diffs:
                    continue
                    
                # Valid!
                row.append(x)
                internal_diffs.update(new_diffs)
                found = True
                break
        else:
            # Random
            # Sample from range(last+1, max_val+1)
            # To be efficient, we can't just pick one if max_val is large.
            # Let's pick a few random candidates.
            candidates = random.sample(range(last + 1, min(last + 50, max_val + 1)), min(10, max(1, min(last + 50, max_val + 1) - (last + 1))))
            for x in candidates:
                new_diffs = {x - a for a in row}
                if len(new_diffs) != len(row):
                    continue
                if new_diffs & used_diffs:
                    continue
                if new_diffs & internal_diffs:
                    continue
                row.append(x)
                internal_diffs.update(new_diffs)
                found = True
                break
        
        if not found:
            return None
            
    return row

def _greedy_construct(n_rows, k):
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
    return rows

def _get_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows, n_rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(row) for row in rows)

def _refine_solution(rows, n_rows, k, deadline):
    """
    Refine a valid solution to reduce the maximum entry.
    """
    current_rows = [row[:] for row in rows]
    current_score = _max_entry(current_rows)
    used_diffs = set()
    for row in current_rows:
        used_diffs.update(_get_diffs(row))
        
    # Iteratively try to improve
    # 1. Find the row with the max entry.
    # 2. Try to replace it with a tighter row.
    # 3. Try to shift elements in other rows to free up space? (Harder)
    
    for _ in range(1000):
        if time.time() > deadline - 5:
            break
            
        # Find row with max entry
        max_row_idx = max(range(n_rows), key=lambda i: max(current_rows[i]))
        max_val = max(current_rows[max_row_idx])
        
        if max_val <= 105: # Counting bound, can't do better
            break
            
        # Try to replace the max row with a tighter one
        # We want a row with max < max_val that fits in the remaining diffs.
        # Remaining diffs = all diffs not in (used_diffs - diffs_of_max_row)
        
        old_row = current_rows[max_row_idx]
        old_diffs = _get_diffs(old_row)
        available_diffs = used_diffs - old_diffs
        
        # Try to generate a row with max < max_val
        # Use backtracking or random generation with strict upper bound
        new_row = None
        for _ in range(50):
            # Try to generate a row with max value < max_val
            # We can use _generate_row_with_constraints but with a tighter max_val
            candidate = _generate_row_with_constraints(k, available_diffs, max_val=max_val - 1)
            if candidate:
                # Check if it's actually tighter
                if max(candidate) < max_val:
                    new_row = candidate
                    break
        
        if new_row:
            # Accept the new row
            current_rows[max_row_idx] = new_row
            new_diffs = _get_diffs(new_row)
            used_diffs = used_diffs - old_diffs
            used_diffs.update(new_diffs)
            current_score = _max_entry(current_rows)
            
        else:
            # Try local search: shift elements in the max row
            # Try to reduce the last element
            row = current_rows[max_row_idx]
            improved = False
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                # Try smaller values
                for new_val in range(old_val - 1, row[pos-1], -1):
                    test_row = row[:]
                    test_row[pos] = new_val
                    if not _is_row_valid_fast(test_row):
                        continue
                    test_diffs = _get_diffs(test_row)
                    # Check conflict with other rows
                    if test_diffs & (used_diffs - _get_diffs(row)):
                        continue
                    # Accept
                    current_rows[max_row_idx] = test_row
                    used_diffs = used_diffs - _get_diffs(row)
                    used_diffs.update(test_diffs)
                    current_score = _max_entry(current_rows)
                    improved = True
                    break
                if improved:
                    break
                    
            if not improved:
                # No improvement possible in this step, break to avoid infinite loop
                break

    return current_rows, current_score

def _is_row_valid_fast(row):
    """Check if a single row is valid (increasing, starts at 0, distinct internal diffs)."""
    if len(row) < 2 or row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    diffs = _get_diffs(row)
    expected = len(row) * (len(row) - 1) // 2
    return len(diffs) == expected

# EVOLVE-BLOCK-END
