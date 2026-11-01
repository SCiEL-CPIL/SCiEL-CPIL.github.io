"""
Strategy: Conflict-Driven Backtracking Search with Local Row Replacement.

The current program uses Simulated Annealing which explores the space broadly but
inefficiently for this specific constraint satisfaction problem. The new strategy:
1. Constructs rows one by one using a backtracking search with a Most Constrained
   Variable (MRV) heuristic. It prioritizes choosing the next element that leaves
   the most "room" for future rows (by checking which differences are already used).
2. Uses a local search phase where it attempts to replace any row with a "better"
   row (lower max entry) that is valid with respect to the other fixed rows.
   This is more targeted than SA moves.
"""
import math
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    deadline = time.time() + 140.0  # 10s safety margin from 150s limit
    
    # 1. Initial Construction using Backtracking with Heuristics
    # We try to build the set from scratch, row by row.
    # To allow flexibility, we shuffle the order in which we "commit" to rows?
    # No, rows are symmetric in the constraint (all diffs distinct), but the *values* 
    # matter for the max entry. 
    # Actually, since we want to minimize the GLOBAL max entry, it's better to 
    # construct rows that are "compact".
    
    # Strategy A: Greedy Backtracking
    # We maintain a set of 'used_diffs'.
    # We generate rows one by one.
    # For each row, we use a backtracking search to find a valid row 
    # that minimizes its own max entry, given the current used_diffs.
    
    best_solution = None
    best_score = float('inf')
    
    # Try multiple restarts with different random seeds/orderings to escape local minima
    # The "ordering" here is just randomizing the search within the backtracker.
    
    for restart in range(50): # Limit restarts to ensure time budget
        if time.time() > deadline - 10:
            break
            
        current_rows = []
        used_diffs = set()
        success = True
        
        # We will construct n_rows rows
        # To improve diversity, we can randomize the "preference" of the backtracker
        # or just rely on random tie-breaking.
        
        for r in range(n_rows):
            row = _backtrack_construct_row(k, used_diffs, deadline - 5, restart_seed=restart*1000+r)
            if row is None:
                success = False
                break
            current_rows.append(row)
            # Update used_diffs
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        if success:
            # Local Search: Try to improve
            improved_rows, improved_score = _local_row_replacement(current_rows, n_rows, k, deadline - 5)
            
            if improved_score < best_score:
                best_score = improved_score
                best_solution = improved_rows
                # If we hit the theoretical lower bound (105), we can stop early
                if best_score <= 105:
                    break
        
        # If we found a valid solution, we can also try to polish it further
        # but the loop above handles restarts.
        
    if best_solution is None:
        # Fallback: Simple Greedy (should always work, though maybe suboptimal)
        return _greedy_construct(n_rows, k)
        
    return best_solution

def _backtrack_construct_row(k, used_diffs, deadline, restart_seed=0):
    """
    Constructs a single row of k+1 integers starting at 0.
    Uses backtracking to find a valid row (internal diffs distinct, no conflict with used_diffs).
    Heuristic: At each step, try candidate values in a randomized order, 
    but prefer smaller values to keep the row compact.
    """
    random.seed(restart_seed)
    
    # We want to find a row [0, x1, x2, ..., xk]
    # x1 < x2 < ... < xk
    # All diffs xj - xi are distinct and not in used_diffs.
    
    # To speed up, we can limit the search space.
    # The max entry in the best known is 112. 
    # If we are constructing row r, the max entry shouldn't exceed ~120 usually.
    # Let's set a soft limit.
    
    def backtrack(pos, current_row):
        if time.time() > deadline:
            return None
            
        if pos == k:
            return current_row[:]
            
        start_val = current_row[-1] + 1
        # Upper bound: Let's try up to 150 initially, expand if needed?
        # For the first few rows, small values are likely.
        # For later rows, we might need larger.
        # Let's try a range.
        
        # Optimization: Generate candidates and sort by some heuristic?
        # Simple approach: Try values in increasing order, but with some randomization 
        # to avoid getting stuck in the same local minima on different restarts?
        # Actually, for a single row, deterministic increasing order is usually best 
        # for minimizing max entry. Randomization helps if we want diversity in the SET.
        
        # Let's try increasing order first, but if we fail, we might need to backtrack.
        # To add diversity, we can shuffle the candidates slightly or use a random start offset?
        # No, the row must start at 0.
        
        # Let's just try increasing order. The "restart" diversity comes from 
        # the fact that previous rows were different, so `used_diffs` is different.
        
        for val in range(start_val, 200): # Hard cap 200 for speed, should be enough
            new_diffs = {val - a for a in current_row}
            
            # Check internal validity (distinct within new_diffs)
            if len(new_diffs) != len(current_row):
                continue
                
            # Check against global used_diffs
            if new_diffs & used_diffs:
                continue
                
            # Check internal distinctness of the row itself (handled by len check above? 
            # Wait. current_row has distinct diffs. new_diffs are diffs from val to all prev.
            # We need to ensure new_diffs don't contain duplicates among themselves.
            # len(new_diffs) == len(current_row) ensures that.
            # We also need to ensure new_diffs don't intersect with existing diffs of current_row?
            # Yes! The row must have ALL distinct internal differences.
            existing_row_diffs = set()
            for i in range(len(current_row)):
                for j in range(i+1, len(current_row)):
                    existing_row_diffs.add(current_row[j] - current_row[i])
            
            if new_diffs & existing_row_diffs:
                continue
            
            # Valid move
            new_row = current_row + [val]
            result = backtrack(pos + 1, new_row)
            if result:
                return result
                
        return None

    return backtrack(1, [0])

def _local_row_replacement(rows, n_rows, k, deadline):
    """
    Local search: Try to replace each row with a "better" one.
    Better = lower max entry, or same max entry but frees up "small" diffs?
    Actually, just lower max entry is the primary goal.
    
    For each row i:
      1. Remove its diffs from the global used set.
      2. Try to find a new row for position i that is valid with the remaining rows.
      3. Prefer rows with lower max entry.
      4. If found and better, replace.
    """
    current_rows = [row[:] for row in rows]
    current_score = max(max(row) for row in current_rows)
    
    # Precompute diffs for each row
    row_diffs = [_get_diffs(row) for row in current_rows]
    
    # We will iterate a few times to allow chains of improvements
    for iteration in range(20):
        if time.time() > deadline:
            break
            
        improved = False
        
        # Shuffle row order to avoid bias
        order = list(range(n_rows))
        random.shuffle(order)
        
        for i in order:
            old_row = current_rows[i]
            old_diffs = row_diffs[i]
            
            # Remove old row's diffs from the "available" pool concept
            # The "used" diffs are the union of all OTHER rows' diffs.
            other_diffs = set()
            for j in range(n_rows):
                if j != i:
                    other_diffs.update(row_diffs[j])
            
            # Try to find a new row for position i
            # We want a row with max entry < current max entry (or at least <= old max)
            # Let's try to find the *smallest* possible valid row.
            
            # We can use the backtracker again, but this time we have a tighter constraint:
            # The new row must not conflict with `other_diffs`.
            
            # To find the "smallest", we can modify the backtracker to stop at the first valid solution
            # if we search in increasing order.
            
            # However, we might want to try multiple candidates if the first one isn't strictly better?
            # For now, let's just find *a* valid row. If its max is smaller, we take it.
            # To be more aggressive, we can try to find a row with max < current_score.
            
            # Let's use a specialized backtracker that limits the max value.
            # Limit = current_score - 1 (to ensure improvement) or old_row[-1] - 1?
            # If we can't improve the global max, maybe we can improve this row's max to free up space?
            # Let's try to find any valid row first.
            
            new_row = _backtrack_construct_row_limited(k, other_diffs, deadline - 1, max_val_limit=current_score - 1)
            
            if new_row is not None:
                new_diffs = _get_diffs(new_row)
                # It's valid by construction.
                # Check if it's better
                if new_row[-1] < old_row[-1]:
                    current_rows[i] = new_row
                    row_diffs[i] = new_diffs
                    current_score = max(max(row) for row in current_rows)
                    improved = True
                    # Update other_diffs for subsequent rows in this iteration?
                    # The loop recalculates other_diffs for each i, so it's fine.
                    
            # Also try to find a row with the SAME max but different diffs?
            # This might help escape local minima where a row is "stuck" at the max.
            # But for now, let's stick to strict improvement.
            
        if not improved:
            break
            
    return current_rows, current_score

def _backtrack_construct_row_limited(k, used_diffs, deadline, max_val_limit=150):
    """
    Same as _backtrack_construct_row but respects max_val_limit.
    """
    def backtrack(pos, current_row):
        if time.time() > deadline:
            return None
            
        if pos == k:
            return current_row[:]
            
        start_val = current_row[-1] + 1
        end_val = min(max_val_limit, 200) # Hard cap
        
        if start_val > end_val:
            return None
            
        for val in range(start_val, end_val + 1):
            new_diffs = {val - a for a in current_row}
            
            if len(new_diffs) != len(current_row):
                continue
                
            if new_diffs & used_diffs:
                continue
                
            existing_row_diffs = set()
            for i in range(len(current_row)):
                for j in range(i+1, len(current_row)):
                    existing_row_diffs.add(current_row[j] - current_row[i])
            
            if new_diffs & existing_row_diffs:
                continue
            
            new_row = current_row + [val]
            result = backtrack(pos + 1, new_row)
            if result:
                return result
                
        return None

    return backtrack(1, [0])

def _get_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

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

# EVOLVE-BLOCK-END
