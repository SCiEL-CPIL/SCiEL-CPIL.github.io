import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 145.0
    m = k + 1  # entries per row
    # Total number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Counting bound: sqrt(2 * total_diffs)
    counting_bound = math.ceil(math.sqrt(2 * total_diffs))
    
    # Initial greedy solution to get an upper bound
    def greedy_solve(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(k):
                found = False
                # Try random candidates in increasing preference for small values
                for attempt in range(1000):
                    # Bias towards smaller values
                    # The gap between consecutive elements should be small on average
                    # but we need to ensure uniqueness of differences.
                    # Let's try values close to the previous one first, then expand.
                    # Actually, a better heuristic for difference sets is often to pick
                    # values that are "spread out" but not too much.
                    # Let's stick to a simple random search with bias.
                    hi = row[-1] + 1 + int(50 * (rng.random() ** 0.5))
                    if hi < row[-1] + 1:
                        hi = row[-1] + 1
                    x = rng.randint(row[-1] + 1, hi)
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        found = True
                        break
                if not found:
                    # Sequential fallback
                    x = row[-1]
                    while True:
                        x += 1
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            break
            rows.append(row)
        return rows

    best_rows = greedy_solve(42)
    best_scope = max(max(r) for r in best_rows)
    
    # If greedy is already at counting bound, we are done (likely)
    if best_scope <= counting_bound:
        return best_rows

    # Cell-wise backtracking with MRV (Minimum Remaining Values)
    # We model the problem as filling a grid of size n_rows x m.
    # row[i][j] is the j-th element of the i-th row.
    # Constraints:
    # 1. row[i][0] == 0 for all i
    # 2. row[i][j] > row[i][j-1] for j > 0
    # 3. All differences row[i][j] - row[i][l] (j > l) are unique across all i, j, l.
    
    # To use MRV, we need to determine the set of possible values for each cell.
    # This is dynamic.
    # We will maintain:
    # - current_grid: 2D list of ints, -1 if unassigned
    # - used_diffs: set of all differences used so far
    # - For each unassigned cell (i, j), the domain is [current_grid[i][j-1] + 1, current_scope]
    #   (Note: we must also ensure that differences with previous elements in the same row are not in used_diffs)
    
    # To make this efficient, we will iterate through cells in an order determined by MRV.
    # However, calculating the exact MRV (number of valid candidates) for every unassigned cell
    # at every step can be expensive. 
    # Approximation: The "difficulty" of a cell (i, j) is roughly proportional to the number of 
    # previous elements in row i (which is j). Cells with larger j have more constraints from 
    # the same row. Also, cells in rows that are already "tight" (large previous values) are harder.
    
    # Let's implement a recursive solver that picks the next cell to fill.
    
    def backtrack_solve(M, seed, time_budget):
        rng = random.Random(seed)
        start = time.time()
        
        # Grid initialization
        # grid[i][j]
        grid = [[-1 for _ in range(m)] for _ in range(n_rows)]
        for i in range(n_rows):
            grid[i][0] = 0
            
        used_diffs = set()
        
        # List of unassigned cells
        # We need a way to pick the next cell.
        # We will maintain a list of unassigned cells (i, j) where j > 0.
        # Initially, all (i, 1) are unassigned.
        
        # To implement MRV efficiently, we will re-evaluate the "cost" of each unassigned cell.
        # Cost = number of valid candidates in [grid[i][j-1]+1, M] that don't conflict with used_diffs or existing grid values.
        
        def get_candidates(i, j, M):
            # Returns list of valid values for grid[i][j]
            # Must be > grid[i][j-1]
            # Must be <= M
            # Differences with grid[i][l] for l < j must not be in used_diffs
            # Also, differences between grid[i][j] and grid[i][l] must be unique among themselves? 
            # No, the constraint is that ALL differences in the final grid are unique.
            # So, for a candidate x at (i, j):
            # 1. x > grid[i][j-1]
            # 2. x <= M
            # 3. For all l < j: (x - grid[i][l]) not in used_diffs
            # 4. The set of new differences {x - grid[i][l] for l < j} must be disjoint from each other?
            #    Yes, because if x - grid[i][l1] == x - grid[i][l2], then grid[i][l1] == grid[i][l2], 
            #    which is impossible since the row is strictly increasing. So they are distinct.
            #    But they must not be in used_diffs.
            
            last_val = grid[i][j-1]
            candidates = []
            # Optimization: Iterate from last_val + 1 to M
            # But M can be large. 
            # However, for small M (near counting bound), the range is small.
            # If M is large, we might need to be smarter.
            # Let's just iterate.
            
            # Pre-compute differences with previous elements?
            # prev_vals = grid[i][0:j]
            
            for x in range(last_val + 1, M + 1):
                ok = True
                # Check against used_diffs
                for l in range(j):
                    d = x - grid[i][l]
                    if d in used_diffs:
                        ok = False
                        break
                if ok:
                    candidates.append(x)
            return candidates

        # To avoid recalculating candidates for all cells at every step (which is O(n_rows * k * M * k)),
        # we will use a heuristic order.
        # Heuristic: Fill cells in increasing order of j (column by column).
        # Within a column, pick the row i that has the smallest number of candidates (MRV).
        
        # Let's try a simpler approach first: Column-by-column, Row-by-row, but with backtracking.
        # This is essentially what the current code does, but row-by-row.
        # Column-by-column might be better because constraints from previous columns are "global" 
        # in a sense? No, constraints are per-row for differences.
        # Actually, differences are only within a row. 
        # So, filling row by row is natural because constraints don't cross rows.
        # Wait, the constraint is "all differences ... are distinct OVER ALL ROWS".
        # So yes, constraints DO cross rows.
        
        # If we fill row by row, when we finish row 0, we have added ~28 diffs.
        # When we start row 1, we have 28 used diffs.
        # If we fill column by column:
        # Col 0: all 0.
        # Col 1: Pick x_01, x_11, x_21, x_31.
        # When picking x_11, we must ensure x_11 - 0 is not in {x_01 - 0}.
        # This couples the rows early on.
        
        # Let's implement Column-by-Column with MRV within the column.
        
        # State:
        # grid
        # used_diffs
        # current_col: j
        
        # If current_col == m, success.
        # Else, for all rows i, compute candidates for (i, current_col).
        # If any row has 0 candidates, fail.
        # Pick row i with minimum candidates (MRV).
        # For each candidate x in candidates[i]:
        #    Assign grid[i][current_col] = x
        #    Update used_diffs: add {x - grid[i][l] for l < current_col}
        #    Recurse to next state.
        #    Note: The next state is not simply current_col+1.
        #    We have filled one cell in column current_col.
        #    We need to fill the remaining cells in column current_col before moving to current_col+1?
        #    Or can we jump ahead?
        #    If we jump ahead, we might have unassigned cells in previous columns? No, we process columns in order.
        #    So, if we fill (i, j), we still have (i', j) for i' != i unassigned.
        #    So the "next" step is to pick another cell in the same column, or if all in column j are filled, move to j+1.
        
        # So the state is: set of unassigned cells.
        # To make it simple, we will maintain a list of unassigned cells.
        # Initially: [(i, 1) for i in range(n_rows)]
        # At each step:
        #   Pick cell (i, j) from unassigned with MRV.
        #   Try candidates.
        
        unassigned = [(i, 1) for i in range(n_rows)]
        
        def solve():
            if time.time() - start > time_budget:
                return False
            
            if not unassigned:
                return True
            
            # Calculate MRV for each unassigned cell
            # To speed up, we can just pick the first one? No, MRV is key.
            # But calculating candidates for all is expensive.
            # Let's try a heuristic: Pick the cell with the largest j (most constrained by row history)?
            # Or the one with the smallest domain size?
            
            # Let's compute domain sizes for a subset or all.
            # Since n_rows is small (4) and k is small (7), unassigned size is at most 28.
            # Calculating candidates for 28 cells * M values * k checks is 28 * 100 * 7 ~ 20k ops.
            # This is fast enough.
            
            best_cell = None
            best_candidates = None
            min_size = float('inf')
            
            # Shuffle unassigned to break ties randomly
            rng.shuffle(unassigned)
            
            for (i, j) in unassigned:
                cands = get_candidates(i, j, M)
                if len(cands) < min_size:
                    min_size = len(cands)
                    best_cell = (i, j)
                    best_candidates = cands
                    if min_size == 0:
                        return False
                    if min_size == 1:
                        # Can't get better than 1, but maybe we should check others?
                        # No, 1 is the minimum possible non-zero.
                        # We can break early if we want, but let's check all to be sure?
                        # Actually, if one has 1, it's the best.
                        # But what if another has 0? We already returned False if 0.
                        # So 1 is the best.
                        # We can break.
                        pass 
                
            if best_cell is None:
                return False
                
            i, j = best_cell
            
            # Remove from unassigned
            unassigned.remove(best_cell)
            
            # Try candidates
            # Sort candidates? Maybe randomize to avoid bias?
            # Let's try smaller values first to keep scope small?
            # But we are fixing M. So we just need ANY valid solution.
            # Randomizing might help escape local minima in the search tree?
            # Let's shuffle candidates.
            rng.shuffle(best_candidates)
            
            for x in best_candidates:
                # Assign
                grid[i][j] = x
                
                # Update used_diffs
                new_diffs = []
                for l in range(j):
                    d = x - grid[i][l]
                    new_diffs.append(d)
                    used_diffs.add(d)
                    
                if solve():
                    return True
                    
                # Undo
                for d in new_diffs:
                    used_diffs.discard(d)
                grid[i][j] = -1
                
            # Restore to unassigned
            unassigned.append(best_cell)
            
            return False
            
        if solve():
            return grid
        return None

    # Binary search on scope
    lo = counting_bound
    hi = best_scope
    found_scope = hi
    found_rows = best_rows

    # Try to find a smaller scope
    while lo <= hi and time.time() < deadline - 10:
        mid = (lo + hi) // 2
        success = False
        # Try a few random seeds for each scope
        for seed in range(3):
            time_left = deadline - time.time() - 10
            if time_left < 1:
                break
            time_budget = min(5.0, time_left / 3)
            grid = backtrack_solve(mid, seed, time_budget)
            if grid is not None:
                # Convert grid to rows
                rows = [grid[i] for i in range(n_rows)]
                # Validate
                seen = set()
                valid = True
                for r in rows:
                    if len(r) != m or r[0] != 0:
                        valid = False
                        break
                    for i in range(1, len(r)):
                        if r[i] <= r[i-1]:
                            valid = False
                            break
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            d = r[j] - r[i]
                            if d in seen:
                                valid = False
                                break
                            seen.add(d)
                    if not valid:
                        break
                if valid:
                    success = True
                    found_scope = mid
                    found_rows = rows
                    break
        
        if success:
            hi = mid - 1
        else:
            lo = mid + 1

    if found_scope < best_scope:
        best_scope = found_scope
        best_rows = found_rows

    return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
