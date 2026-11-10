"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

sys.setrecursionlimit(100000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # 1. Generate a valid baseline using a fast greedy method
    # This ensures we always have a valid solution to return if search fails
    baseline_rows = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline_rows) if baseline_rows else 0
    
    # 2. Determine search bounds
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Theoretical lower bound: M must be at least total_diffs
    # Because there are M possible difference values (1..M) and we need total_diffs distinct ones.
    lower_bound = total_diffs
    # For the specific task (7,5), lower_bound is 105. Best known is 112.
    
    # We search M from lower_bound upwards.
    # If we find a solution, we return it immediately as it's the minimal M.
    # If we run out of time, we return the best valid solution found (baseline or partial).
    
    # Optimization: Precompute bit masks for differences 1..M?
    # M is dynamic. We will compute bits on the fly or use 1 << d.
    
    best_rows = baseline_rows
    best_scope = baseline_scope
    
    # Search for minimal M
    # We start from lower_bound. 
    # If the problem is hard, we might not find it quickly.
    # We will try a few random restarts for each M if deterministic fails.
    
    rng = random.Random(42)
    
    # To be safe, if the baseline is already very good (close to lower bound), 
    # we might just return it if the search takes too long.
    
    for M in range(lower_bound, baseline_scope + 1):
        if time.time() > deadline:
            break
            
        # Try to solve with scope M
        # We allow multiple attempts with different randomization strategies
        # to handle cases where deterministic order gets stuck.
        
        found = False
        # Deterministic first
        sol = solve_with_scope(n_rows, k, M, rng, deterministic=True)
        if sol is not None:
            return sol
        
        # Randomized restarts
        # Limit restarts per M to avoid wasting time on impossible M values
        max_restarts = 10
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
            sol = solve_with_scope(n_rows, k, M, rng, deterministic=False)
            if sol is not None:
                return sol
                
    return best_rows

def greedy_construct(n_rows, k):
    """
    Fast greedy construction to get a valid baseline.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def solve_with_scope(n_rows, k, M, rng, deterministic=True):
    """
    Backtracking solver to find n_rows rows of size k+1 with max element <= M,
    such that all pairwise differences are distinct.
    """
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Bitmask for used differences
    used_mask = 0
    rows = []
    
    # Precompute bit values for differences 1..M to speed up operations?
    # 1 << d is fast enough in Python for d < 200.
    
    def dfs(current_row, current_row_mask, rows_completed):
        nonlocal used_mask, rows
        
        # Check time
        if time.time() > deadline:
            return False
            
        # Pruning: Check if enough differences are available
        # used_mask contains diffs from completed rows
        # current_row_mask contains diffs from the current row
        used_so_far = used_mask | current_row_mask
        available_diffs = M - used_so_far.bit_count()
        
        # Diffs needed:
        # 1. Complete current row: diffs_per_row - current_row_mask.bit_count()
        # 2. Remaining rows: (n_rows - rows_completed - 1) * diffs_per_row
        needed_diffs = (diffs_per_row - current_row_mask.bit_count()) + (n_rows - rows_completed - 1) * diffs_per_row
        
        if available_diffs < needed_diffs:
            return False
            
        # Base case: Current row is complete
        if len(current_row) == k + 1:
            # Add to completed rows
            rows.append(list(current_row))
            used_mask |= current_row_mask
            
            # If all rows are complete, success
            if rows_completed + 1 == n_rows:
                return True
            
            # Recurse for next row
            if dfs([0], 0, rows_completed + 1):
                return True
            
            # Backtrack
            rows.pop()
            # Note: used_mask is a nonlocal int, so we need to restore it
            # Since we did used_mask |= current_row_mask, we can't easily "un-or" 
            # unless we saved the previous state.
            # Better approach: Pass used_mask as a local variable in recursion?
            # Or save/restore.
            # Let's refactor to pass used_mask as argument to avoid global state mutation issues.
            return False
        
        # Extend current row
        last_val = current_row[-1]
        candidates = []
        
        # Candidate x must be > last_val and <= M
        # Also, the smallest possible next value is last_val + 1
        # The largest possible next value is M
        
        for x in range(last_val + 1, M + 1):
            new_diffs_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                bit = 1 << d
                # Check if this diff is already used in completed rows or current row
                if (used_mask | current_row_mask) & bit:
                    valid = False
                    break
                new_diffs_mask |= bit
            
            if valid:
                candidates.append((x, new_diffs_mask))
                
        if not candidates:
            return False
            
        if not deterministic:
            rng.shuffle(candidates)
            
        for x, new_diffs_mask in candidates:
            current_row.append(x)
            # Recurse
            if dfs(current_row, current_row_mask | new_diffs_mask, rows_completed):
                return True
            current_row.pop()
            
        return False

    # Wrapper to handle the global state correctly
    # We need to be careful with `used_mask` and `rows` being nonlocal.
    # Let's rewrite the core to use local variables passed by reference (lists) or return values.
    # Actually, the previous implementation had a flaw: `used_mask` is modified in place, 
    # but not restored on backtrack in the base case.
    
    # Let's use a cleaner recursive structure with local state.
    
    def dfs_clean(current_row, current_row_mask, used_mask_so_far, rows_so_far):
        if time.time() > deadline:
            return None
        
        # Pruning
        used_combined = used_mask_so_far | current_row_mask
        available_diffs = M - used_combined.bit_count()
        needed_diffs = (diffs_per_row - current_row_mask.bit_count()) + (n_rows - len(rows_so_far) - 1) * diffs_per_row
        if available_diffs < needed_diffs:
            return None
        
        if len(current_row) == k + 1:
            new_rows = rows_so_far + [list(current_row)]
            new_used_mask = used_mask_so_far | current_row_mask
            
            if len(new_rows) == n_rows:
                return new_rows
            
            # Recurse for next row
            res = dfs_clean([0], 0, new_used_mask, new_rows)
            return res
        
        last_val = current_row[-1]
        candidates = []
        
        for x in range(last_val + 1, M + 1):
            new_diffs_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                bit = 1 << d
                if used_combined & bit:
                    valid = False
                    break
                new_diffs_mask |= bit
            
            if valid:
                candidates.append((x, new_diffs_mask))
                
        if not candidates:
            return None
            
        if not deterministic:
            rng.shuffle(candidates)
            
        for x, new_diffs_mask in candidates:
            current_row.append(x)
            res = dfs_clean(current_row, current_row_mask | new_diffs_mask, used_mask_so_far, rows_so_far)
            if res is not None:
                return res
            current_row.pop()
            
        return None

    return dfs_clean([0], 0, 0, [])

# EVOLVE-BLOCK-END
