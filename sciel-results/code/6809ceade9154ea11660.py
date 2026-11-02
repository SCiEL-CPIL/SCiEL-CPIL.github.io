import random
import time
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set (n_rows, k).
    Each row has k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    # Total number of differences to be generated
    # Each row of size k+1 has (k+1)*k/2 differences.
    # Total diffs = n_rows * (k+1)*k/2
    total_diffs = n_rows * (k + 1) * k // 2

    best_scope = float('inf')
    best_solution = None

    # Helper to check if a candidate mark x for row r, position p is valid
    # It must not create a duplicate difference with:
    # 1. Previous marks in the same row
    # 2. Any marks in any previous rows (differences with those marks)
    # Note: We build rows sequentially. Within a row, we build marks sequentially.
    
    def try_solve(seed):
        nonlocal best_scope, best_solution
        rng = random.Random(seed)
        
        # State:
        # rows: list of lists. rows[i] is the current row i.
        # used_diffs: set of all differences used so far.
        # For efficiency, we might track used diffs per row or globally.
        # Since the constraint is global uniqueness of all differences, a global set is correct.
        
        # We will use backtracking.
        # State: (row_idx, mark_idx, current_row, used_diffs)
        # To avoid copying used_diffs (expensive), we can use a "undo" stack or just copy if depth is small.
        # Given k is likely small (e.g., 7 or 8 in typical tasks, but here k is param), 
        # and n_rows is small, the total marks is n_rows*(k+1).
        # If k=7, marks per row=8. Total marks for 4 rows = 32.
        # used_diffs size ~ 4*28 = 112.
        # Copying a set of size 100-1000 is fast in Python.
        
        def backtrack(row_idx, mark_idx, current_row, used_diffs):
            nonlocal best_scope, best_solution
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                # Found a solution
                scope = max(max(row) for row in rows_state)
                if scope < best_scope:
                    best_scope = scope
                    best_solution = [row[:] for row in rows_state]
                return True
            
            if mark_idx == len(current_row) and len(current_row) < k + 1:
                # Need to add more marks to current row
                # Try to find the next mark
                # Heuristic: Try smallest possible x first to keep scope small?
                # Or randomize?
                # Let's try a mix: start from 1, but with some randomness in step or restarts.
                # Actually, for minimizing max, greedy smallest is usually good, but backtracking allows recovery.
                
                # Optimization: We can bound the search for x.
                # x must be > current_row[-1]
                # Also, any new difference x - a for a in current_row must be in used_diffs.
                # Also, differences with previous rows must be in used_diffs.
                
                # Let's iterate x from current_row[-1] + 1 upwards.
                # To improve on greedy, we can allow "jumps" or randomize the order of x candidates?
                # No, x must be increasing. So we must try x=last+1, last+2, etc.
                # But we can limit how far we go?
                # If we go too far, scope increases.
                # We can prune if current_row[-1] + 1 already leads to a scope > best_scope? 
                # Not strictly, because later marks might be smaller? No, marks are increasing.
                # So if current_row[-1] > best_scope, we can prune this branch.
                
                # Let's try candidates x.
                # To add variety, we can shuffle the *order* of rows? 
                # The task says "Choose 4 rows...". The rows are distinguishable by content, but the set of rows is what matters.
                # However, the order in which we construct them might matter for the search space.
                # Let's stick to row 0, then row 1, etc.
                
                # Improvement: Instead of just linear scan, we can use a heuristic to skip x that are "obviously" bad?
                # Or just rely on backtracking.
                
                # Let's try x from last+1 to some limit.
                # Limit: best_scope (if we have a solution) or a large number.
                # If best_scope is inf, we just go.
                
                last = current_row[-1]
                # We can also try to "restart" the row construction with different initial steps?
                # No, the first mark is 0. The second is x1.
                
                # Let's add randomization to the search:
                # Occasionally, skip some small x values to explore larger x values earlier?
                # This might help escape local minima where small x's block future progress.
                
                # Strategy: 
                # 1. Try x = last + 1, last + 2, ... 
                # 2. But if we fail, backtrack.
                # 3. To speed up, we can limit the range of x if we have a good best_scope.
                
                limit = best_scope if best_scope < float('inf') else last + 1000
                # Actually, if best_scope is 134, we only care about x <= 134.
                
                # To introduce randomness, we can pick a random subset of candidates or randomize the step?
                # Let's try a "randomized greedy" inside the backtracking:
                # Pick a random x from a window [last+1, last+window_size] or up to limit?
                # No, we need to find *a* valid x.
                
                # Let's iterate x.
                for x in range(last + 1, limit + 1):
                    new_diffs = {x - a for a in current_row}
                    # Check against used_diffs
                    if new_diffs & used_diffs:
                        continue
                    
                    # Check against differences with previous rows
                    # For each previous row r_prev, for each a in r_prev:
                    # diff = abs(x - a) must not be in used_diffs?
                    # Wait, used_diffs contains ALL differences from ALL previous rows.
                    # So if we maintain used_diffs as the set of ALL diffs from completed rows + current row so far,
                    # then checking `new_diffs & used_diffs` is sufficient for internal row conflicts.
                    # But we also need to check conflicts with previous rows.
                    # Let's maintain `all_used_diffs` which includes diffs from all completed rows.
                    # And `current_row_diffs` which includes diffs within current row.
                    # Actually, simpler: `used_diffs` contains all diffs from rows 0..row_idx-1 AND current_row so far.
                    # When we add x to current_row:
                    # 1. Diffs within current_row: x - a for a in current_row. These must be new (not in used_diffs).
                    # 2. Diffs with previous rows: |x - a| for a in rows[0..row_idx-1]. These must be new (not in used_diffs).
                    
                    # So we need to check both.
                    valid = True
                    # 1. Internal
                    for a in current_row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # 2. External (with previous rows)
                    for r in range(row_idx):
                        for a in rows_state[r]:
                            d = abs(x - a)
                            if d in used_diffs:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # If valid, recurse
                    new_current_row = current_row + [x]
                    new_used_diffs = used_diffs.union(new_diffs)
                    # Add external diffs to new_used_diffs? 
                    # Yes, because they become "used" for subsequent marks in this row and future rows.
                    for r in range(row_idx):
                        for a in rows_state[r]:
                            new_used_diffs.add(abs(x - a))
                    
                    if backtrack(row_idx, mark_idx + 1, new_current_row, new_used_diffs):
                        return True
                        
                    # Backtrack: do nothing, state is passed by value (sets are immutable in this context if we use union)
                    # Wait, sets are mutable. But we created new_used_diffs.
                    # current_row is a list. We created new_current_row.
                    # So we are safe.
                    
                return False
            
            else:
                # Row is complete (len == k+1)
                # Move to next row
                rows_state.append(current_row)
                # used_diffs already contains all diffs for this row.
                return backtrack(row_idx + 1, 0, [0], used_diffs)

        rows_state = []
        # Start with row 0, mark 0 (which is 0)
        # Actually, let's handle the first mark 0 explicitly.
        # Row starts with [0].
        # Diffs from 0 are none yet.
        # So we call backtrack(0, 1, [0], set())
        # Wait, mark_idx 1 means we have 1 mark (0). We need k+1 marks.
        # So we need to add k more marks.
        
        # Optimization: Pre-compute nothing, just run.
        # To make it randomized, we can shuffle the order of x candidates?
        # No, x must be increasing.
        # But we can randomize the *starting point* or *step*?
        # Or use multiple restarts with different random seeds to change the heuristic?
        
        # Let's add a heuristic: Instead of always trying smallest x first,
        # we can try x's in a randomized order within a small window, or just smallest.
        # Smallest x first is generally good for minimizing max.
        # However, it can lead to dead ends. Backtracking handles that.
        
        # To improve search efficiency, we can use "conflict-directed backjumping" or similar,
        # but simple backtracking with good heuristics might suffice for small n, k.
        
        # Let's run a few restarts with different random seeds to potentially find different local optima?
        # Backtracking is deterministic if x order is deterministic.
        # To make it stochastic, we can randomize the x order slightly?
        # E.g., for x in range(last+1, limit+1):
        #    if random() < 0.1: skip? No, that breaks completeness.
        #    Better: shuffle a list of candidates?
        #    Candidates are [last+1, ..., limit].
        #    If we shuffle, we might try large x first, leading to large scope, but maybe valid.
        #    Since we want minimum scope, trying small x first is a good heuristic.
        #    But if small x leads to a dead end deep in the tree, we waste time.
        #    
        #    Let's stick to deterministic x order (smallest first) but use multiple restarts
        #    with different *row orders*?
        #    The rows are symmetric in the problem statement (set of rows).
        #    But the construction order matters.
        #    Let's shuffle the row indices? No, rows are identical in structure.
        #    
        #    Alternative: Randomize the *initial* few marks?
        #    No, mark 0 is fixed.
        
        # Let's just run the deterministic backtracking. If it's too slow, we need to optimize.
        # For n=4, k=7, it should be fast enough.
        
        # Wait, the current program is greedy. Backtracking is strictly more powerful.
        # Let's implement it cleanly.
        
        pass # Placeholder, actual logic below

    # Actual implementation
    def solve():
        nonlocal best_scope, best_solution
        start = time.time()
        
        # State
        rows_state = []
        
        def backtrack(row_idx, current_row, used_diffs):
            nonlocal best_scope, best_solution
            if time.time() - start > time_limit:
                return False
            
            if row_idx == n_rows:
                scope = max(max(row) for row in rows_state)
                if scope < best_scope:
                    best_scope = scope
                    best_solution = [row[:] for row in rows_state]
                return True
            
            if len(current_row) == k + 1:
                rows_state.append(current_row)
                res = backtrack(row_idx + 1, [0], used_diffs)
                rows_state.pop()
                return res
            
            # Need to add a mark
            last = current_row[-1]
            limit = best_scope if best_scope < float('inf') else last + 100
            
            # Heuristic: Try smallest x first
            for x in range(last + 1, limit + 1):
                # Check internal diffs
                valid = True
                new_internal_diffs = set()
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_internal_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check external diffs
                for r in range(row_idx):
                    for a in rows_state[r]:
                        d = abs(x - a)
                        if d in used_diffs:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Recurse
                new_used_diffs = used_diffs.copy()
                new_used_diffs.update(new_internal_diffs)
                for r in range(row_idx):
                    for a in rows_state[r]:
                        new_used_diffs.add(abs(x - a))
                        
                if backtrack(row_idx, current_row + [x], new_used_diffs):
                    return True
                    
            return False
        
        # Start
        backtrack(0, [0], set())
        
    # Run multiple restarts? 
    # The backtracking is deterministic. 
    # To explore different solution spaces, we can randomize the search order of x?
    # Let's try a randomized version: shuffle the candidate x's?
    # No, x must be increasing in the row.
    # But we can randomize the *limit* or *step*?
    
    # Let's try a different approach: 
    # Use the backtracking, but if it gets stuck, restart with a different heuristic?
    # Or just run it once. It should find the optimal or near-optimal.
    
    solve()
    
    if best_solution is None:
        # Fallback to greedy if backtracking fails or times out without finding anything
        # This shouldn't happen for small n, k
        return _greedy_fallback(n_rows, k)
        
    return best_solution

def _greedy_fallback(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                # Check external
                valid = True
                for r in range(len(rows)):
                    for a in rows[r]:
                        if abs(x - a) in used:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    row.append(x)
                    used.update(new)
                    for r in range(len(rows)):
                        for a in rows[r]:
                            used.add(abs(x - a))
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
