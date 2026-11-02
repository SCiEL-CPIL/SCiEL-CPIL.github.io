import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 115.0  # Safety margin for 120s limit
    m = k + 1  # entries per row
    total_diffs_per_row = k * (k + 1) // 2
    
    # Pre-compute all pairs (i, j) within a row for difference calculation
    # diff_index[r][i][j] corresponds to the difference a[r][j] - a[r][i]
    # We need to ensure all these differences are distinct.
    
    # Variables: x[r][p] for r in 0..n_rows-1, p in 0..k
    # Constraints:
    # 1. x[r][0] = 0
    # 2. x[r][p] < x[r][p+1] for all p
    # 3. All differences x[r][j] - x[r][i] (i<j) are distinct across all rows.
    
    # Strategy:
    # 1. Greedy initialization to get a valid upper bound (scope).
    # 2. Backtracking search with MRV to find a solution with a smaller scope.
    #    We iterate target_scope from a lower bound up to the greedy upper bound.
    
    def greedy_init():
        """Fast greedy construction to get a valid solution and upper bound."""
        used_diffs = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for p in range(1, m):
                # Find smallest valid x
                x = row[-1] + 1
                while True:
                    valid = True
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        for i in range(len(row)):
                            used_diffs.add(x - row[i])
                        row.append(x)
                        break
                    x += 1
            rows.append(row)
        return rows, max(max(r) for r in rows)

    init_rows, init_scope = greedy_init()
    best_rows = init_rows
    best_scope = init_scope
    
    # Lower bound for scope
    # Total differences needed: n_rows * total_diffs_per_row
    # Differences are in 1..scope
    # So scope must be at least n_rows * total_diffs_per_row
    lb = n_rows * total_diffs_per_row
    
    # We will try to find a solution with scope < best_scope
    # We iterate scope from max(lb, best_scope - 20) up to best_scope - 1
    # If we find one, we update best_scope and continue down or stop.
    
    # To make backtracking efficient, we fix the order of assignment.
    # We assign variables in a specific order.
    # MRV heuristic: At each step, pick the variable with the fewest available values.
    # However, implementing dynamic MRV with backtracking in Python is complex and slow.
    # Alternative: Fixed order backtracking with pruning.
    # Order: Row 0, then Row 1, etc. Within a row, left to right.
    # Pruning: Check if assigning x[r][p] creates a duplicate difference with any *already assigned* difference.
    # Also check if it violates ordering.
    
    # Optimization: Pre-calculate the "potential" differences.
    # Actually, simple backtracking might be too slow for scope ~130 if not guided well.
    # Let's use a hybrid:
    # 1. Try backtracking for a few seconds.
    # 2. If it fails, fall back to local search on the greedy solution (like the current program but maybe better).
    # But the directive asks for a structural change. The CSP approach is the structural change.
    
    # Let's implement a recursive backtracking solver.
    # State:
    # rows: list of lists, partially filled.
    # used_diffs: set of all differences used so far.
    # current_r, current_p: index of next variable to assign.
    
    # To speed up, we can fix the first row to be [0, 1, 2, ..., k] ?
    # No, that might be too restrictive.
    # But we can try to constrain the search space by targeting a specific scope M.
    
    def backtrack_search(target_scope, time_limit):
        """
        Try to find a valid configuration with max entry <= target_scope.
        Returns (rows, success).
        """
        rows = [[0] * m for _ in range(n_rows)]
        used_diffs = set()
        
        # Variable order: (r, p)
        # We will iterate p from 1 to k for each row r.
        # r from 0 to n_rows-1.
        
        # To improve pruning, we can sort rows? No, rows are distinct.
        # We can use symmetry breaking? If rows are interchangeable, we can enforce row 0 <= row 1 ... lexicographically?
        # This reduces search space by factor of n_rows!.
        # Let's enforce: max(rows[0]) <= max(rows[1]) ... is not simple to enforce during construction.
        # Simpler symmetry breaking: rows[0][1] <= rows[1][1] <= ...
        # Or just rows[0] <= rows[1] element-wise?
        # Let's try enforcing rows[i] <= rows[i+1] element-wise.
        
        def is_valid_assignment(r, p, val):
            # Check ordering in current row
            if p > 0 and val <= rows[r][p-1]:
                return False
            if p < m-1 and val >= rows[r][p+1]: # Only if p+1 is already assigned? No, we assign in order.
                # Since we assign in order p=1..k, rows[r][p+1] is 0 (unassigned) or not set yet.
                # We only check against previous.
                pass
            
            # Check differences with previous elements in this row
            for i in range(p):
                d = val - rows[r][i]
                if d in used_diffs:
                    return False
                # We don't add to used_diffs yet, just check
            return True

        def assign(r, p, val):
            rows[r][p] = val
            for i in range(p):
                d = val - rows[r][i]
                used_diffs.add(d)

        def unassign(r, p):
            for i in range(p):
                d = val - rows[r][i] # val is rows[r][p]
                used_diffs.remove(d)
            rows[r][p] = 0

        # Recursive helper
        # To avoid recursion limit and overhead, we might use iterative or careful recursion.
        # Depth is n_rows * k. For 4*7=28, recursion is fine.
        
        sys.setrecursionlimit(10000)
        
        def dfs(r, p, start_val):
            if time.time() > time_limit:
                return False
            
            if r == n_rows:
                return True
            
            # If p == 0, it's fixed to 0. Move to p=1.
            if p == 0:
                return dfs(r, 1, 1)
            
            # If p == m, move to next row
            if p == m:
                # Symmetry breaking: ensure this row is "less than or equal" to the next row?
                # Hard to enforce without lookahead.
                # Just move to next row.
                return dfs(r+1, 0, 0)
            
            # Determine range for val
            # val must be > rows[r][p-1]
            # val must be <= target_scope
            # Also, if r > 0 and we enforce symmetry, rows[r] should be >= rows[r-1]?
            # Let's try: rows[r][p] >= rows[r-1][p]
            min_val = rows[r][p-1] + 1
            if r > 0:
                min_val = max(min_val, rows[r-1][p])
            
            max_val = target_scope
            
            # Optimization: If we are in the last position of the last row, we must fill it.
            # Also, we can prune if the minimum possible future differences exceed available slots?
            # Too complex. Just iterate.
            
            # To find a solution faster, we might want to try values close to the minimum first (greedy-like)
            # or randomize?
            # For finding *any* solution with scope M, greedy-like (smallest first) is usually good for feasibility.
            # For finding *minimal* M, we binary search M.
            
            for val in range(min_val, max_val + 1):
                # Check validity
                valid = True
                for i in range(p):
                    d = val - rows[r][i]
                    if d in used_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Assign
                assign(r, p, val)
                
                if dfs(r, p+1, val+1):
                    return True
                
                # Unassign
                unassign(r, p)
                
                # If we've spent too much time, abort
                if time.time() > time_limit:
                    return False

        return dfs(0, 0, 0), rows

    # We will try to find a solution with scope < best_scope.
    # We iterate candidate_scope from max(lb, best_scope - 30) to best_scope - 1.
    # We give each candidate a time slice.
    
    # If n_rows * k is small, we can be more aggressive.
    # For 4x7, depth is 28.
    
    # Let's try a few scopes.
    # Start from best_scope - 1 down to lb.
    # But backtracking is exponential. We need a tight time limit per scope.
    
    # Let's try the best_scope - 1 first.
    # If found, update best_scope and try best_scope - 1 again.
    
    current_target = best_scope - 1
    # We can also try to jump down if lb is close.
    
    # To avoid spending all time on one failing scope, we limit time per scope.
    time_per_scope = 10.0
    
    while current_target >= lb and time.time() < deadline - 5:
        # Check if current_target is even possible given counting bound?
        # n_rows * total_diffs_per_row <= current_target
        # This is already handled by lb.
        
        start_time = time.time()
        found, rows_sol = backtrack_search(current_target, start_time + time_per_scope)
        
        if found:
            # Verify
            if is_valid_sol(rows_sol, current_target):
                best_rows = rows_sol
                best_scope = current_target
                # Continue trying to lower
                # We can reduce time_per_scope if we are close to the end
            else:
                # Should not happen if backtrack is correct, but just in case
                break
        else:
            # No solution found in this scope within time limit.
            # It might exist, but we didn't find it.
            # We should probably stop trying lower scopes if we can't even find one at current_target?
            # Actually, if scope M is infeasible, M-1 is also infeasible.
            # If we failed to find a solution at M, it's either infeasible or we timed out.
            # If we timed out, we don't know.
            # If we are confident in the search (e.g. for small sizes), we can assume infeasible.
            # For 4x7, the search space is large.
            # Let's assume if we time out, we can't be sure.
            # But for the purpose of this challenge, if we don't find it quickly, it's likely hard or infeasible.
            # We will stop trying lower scopes if we time out, to preserve the best found so far.
            # Or we can try one more step down? No, that's risky.
            break

    # If backtracking didn't improve, we return the greedy solution.
    # But we can also try a simple local search on the greedy solution to see if we can squeeze it.
    # The current program does this. The new strategy is the CSP.
    # If CSP fails to improve, we return the greedy result.
    
    return best_rows

def is_valid_sol(rows, scope):
    """Verify a solution."""
    used = set()
    n_rows = len(rows)
    k = len(rows[0]) - 1
    for r in range(n_rows):
        row = rows[r]
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if row[i] > scope:
                return False
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

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
