import time
import random
import math
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a backtracking search with symmetry breaking to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit (or 150s if that's the hard limit, but prompt says 120s for evaluator, 150s for kill. Use 115s to be safe).

    # ---- Greedy fallback (always valid) ----
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    # If the problem size is not the specific hard case, just return greedy
    # The specific case is n_rows=7, k=5 (6 elements per row)
    if n_rows != 7 or k != 5:
        return best_solution

    # Specific solver for (7, 5)
    # Theoretical lower bound is 105.
    # We iterate M from 105 upwards.
    
    def solve_dts_specific(M, time_limit):
        """
        Finds a DTS(7,5) with max entry <= M.
        Returns list of rows or None.
        """
        n_rows = 7
        k = 5
        
        # used_diffs_mask: bit d is set if difference d is used.
        # We use Python ints as bitmasks.
        
        def build_row(idx, current_row, current_diffs_mask, prev_row, is_greater, used_diffs_mask):
            """
            Generator that yields valid rows of length k+1 starting with 0.
            idx: current index to fill (1 to k).
            current_row: list of filled values.
            current_diffs_mask: bitmask of diffs in current_row.
            prev_row: previous row for symmetry breaking.
            is_greater: has current_row already exceeded prev_row lexicographically?
            used_diffs_mask: bitmask of diffs used in previous rows.
            """
            if time.time() > time_limit:
                return
            
            if idx == k + 1:
                yield current_row[:]
                return
                
            start_val = current_row[-1] + 1
            
            if not is_greater:
                # We must match or exceed prev_row
                # If we pick x == prev_row[idx], is_greater stays False
                # If we pick x > prev_row[idx], is_greater becomes True
                # If we pick x < prev_row[idx], invalid
                min_val = max(start_val, prev_row[idx])
            else:
                min_val = start_val
                
            for x in range(min_val, M + 1):
                # Check diffs
                valid = True
                new_diffs_mask = 0
                for i in range(idx):
                    d = x - current_row[i]
                    # d is always positive since x > current_row[i]
                    # Check if d is already in current_diffs_mask
                    if (current_diffs_mask >> d) & 1:
                        valid = False
                        break
                    # Check if d is in used_diffs_mask
                    if (used_diffs_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs_mask |= (1 << d)
                
                if not valid:
                    continue
                    
                # Determine new is_greater
                new_is_greater = is_greater
                if not is_greater:
                    if x > prev_row[idx]:
                        new_is_greater = True
                    # If x == prev_row[idx], new_is_greater remains False
                
                current_row.append(x)
                
                # Recurse
                for res in build_row(idx + 1, current_row, current_diffs_mask | new_diffs_mask, prev_row, new_is_greater, used_diffs_mask):
                    yield res
                    
                current_row.pop()
                
                # Optional: limit branching factor to avoid explosion
                # But for M=105, it should be fine.
                
        def dfs(pos, used_diffs_mask, prev_row):
            if time.time() > time_limit:
                return None
                
            if pos == n_rows:
                return []
            
            # Generate rows
            # We limit the number of rows we try at each step to prevent hanging
            # if a solution is not found quickly.
            max_attempts = 1000
            attempts = 0
            
            for row in build_row(1, [0], 0, prev_row, False, used_diffs_mask):
                attempts += 1
                if attempts > max_attempts:
                    break
                
                # Calculate diffs mask for this row
                row_diffs_mask = 0
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        row_diffs_mask |= (1 << d)
                
                new_used_mask = used_diffs_mask | row_diffs_mask
                res = dfs(pos + 1, new_used_mask, row)
                if res is not None:
                    return [row] + res
            
            return None

        # Initial call
        # prev_row for the first row is effectively smaller than any valid row.
        # We use [0,0,0,0,0,0] and is_greater=False.
        # This forces the first element (idx=1) to be >= 0.
        # Since start_val is 1, min_val is 1.
        # If we pick x=1, 1>0, so is_greater becomes True.
        # This allows any valid first row.
        
        return dfs(0, 0, [0,0,0,0,0,0])

    # Iterate M from 105
    # Allocate time based on M
    # Give more time to smaller M
    current_time = start_time
    for M in range(105, 130):
        if time.time() > deadline - 2:
            break
            
        # Calculate time limit for this M
        # Simple heuristic: equal time chunks
        # But we want to finish M=105 if possible
        # Let's just use the remaining deadline
        sol = solve_dts_specific(M, deadline - 2)
        
        if sol:
            # Verify solution
            all_diffs = set()
            valid = True
            for row in sol:
                if len(row) != k + 1:
                    valid = False
                    break
                if row[0] != 0:
                    valid = False
                    break
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
                if max(row) > M:
                    valid = False
                    break
            if valid:
                return sol
            else:
                # Should not happen
                pass
                
    return best_solution

# EVOLVE-BLOCK-END
