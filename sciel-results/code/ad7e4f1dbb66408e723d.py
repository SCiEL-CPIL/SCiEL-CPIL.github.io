import sys
import time
import math

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimize the largest entry.
    """
    total_diffs = n_rows * k * (k + 1) // 2
    M_min = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2))
    
    # Known best for (4,7) is 134. 
    # We start from M_min and go up.
    # To be safe against timeouts, we cap the search time.
    global_start = time.time()
    
    for M in range(M_min, 200):
        if time.time() - global_start > 100:
            break
        sol = _search(M, n_rows, k, global_start)
        if sol:
            return sol
            
    # Fallback
    return _greedy_fallback(n_rows, k)

def _search(M, n_rows, k, start_time):
    # used_diffs[i] is True if difference i is used
    used_diffs = [False] * (M + 1)
    rows = []
    
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def dfs(row_idx, current_row):
        if time.time() - start_time > 105:
            return False
            
        if row_idx == n_rows:
            return True
            
        if len(current_row) == k + 1:
            # Row is complete. Check if all diffs are valid (not used)
            diffs = get_diffs(current_row)
            for d in diffs:
                if d > M or used_diffs[d]:
                    return False
                    
            # Mark them as used
            for d in diffs:
                used_diffs[d] = True
                
            rows.append(current_row)
            
            if dfs(row_idx + 1, [0]):
                return True
            
            # Backtrack: unmark
            for d in diffs:
                used_diffs[d] = False
            rows.pop()
            return False
            
        # Find next element
        last = current_row[-1]
        
        # Pruning: 
        # We need to place (k+1 - len(current_row)) more elements.
        # The max value is M.
        # The smallest possible next value is last + 1.
        
        for x in range(last + 1, M + 1):
            # Check if x is valid
            valid = True
            for a in current_row:
                d = x - a
                if d > M:
                    valid = False
                    break
                if used_diffs[d]:
                    valid = False
                    break
                # Check internal distinctness for this row
                # Since row is sorted, x-a is strictly decreasing as a increases?
                # No, x is fixed. a increases. x-a decreases.
                # So diffs are distinct if x-a are distinct.
                # x-a1, x-a2... are distinct if a1 != a2.
                # So we just need to check against used_diffs and M.
                # Wait, we also need to ensure that the NEW diffs don't conflict with EACH OTHER.
                # Since a_i are distinct, x-a_i are distinct.
                # So we only need to check against global used_diffs.
                
            if valid:
                # Optimization: Check if we can complete the rest of the row?
                # Simple forward check: 
                # We have r spots left.
                # We need r * len(current_row) new diffs? No.
                # Just try.
                if dfs(row_idx, current_row + [x]):
                    return True
                    
        return False

    return dfs(0, [0])

def _greedy_fallback(n_rows, k):
    used_diffs = set()
    rows = []
    for r in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if len(new_diffs) == len(row) and not (new_diffs & used_diffs):
                row.append(x)
                used_diffs |= new_diffs
            x += 1
        rows.append(row)
    return rows
