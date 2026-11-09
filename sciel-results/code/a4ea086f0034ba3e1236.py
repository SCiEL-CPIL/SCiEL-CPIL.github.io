"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import sys
from collections import defaultdict

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry (scope).
    
    Uses a backtracking search with constraint propagation.
    """
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # Try increasing bounds until we find a valid solution
    # Start from a reasonable lower bound
    # The counting bound suggests m >= total_diffs, but we need more
    # Let's try bounds starting from a heuristic estimate
    
    # For (7,5): total_diffs = 7*15 = 105
    # Best known is 112. Let's try from 105 upwards.
    
    def search(m):
        """
        Backtracking search for a valid configuration with max entry <= m.
        Each row is a sequence a_0=0 < a_1 < ... < a_k <= m.
        All differences a_j - a_i (i<j) must be distinct across all rows.
        """
        used_diffs = set()
        rows = []
        
        def propagate(row_idx, pos, current_row):
            """
            Check if we can continue building row row_idx at position pos.
            Returns list of candidate values for current_row[pos], or None if dead end.
            current_row has len = pos+1 (indices 0..pos), with current_row[pos] to be determined.
            Actually, let's say current_row has the first pos elements fixed, and we're choosing element at index pos.
            So current_row = [a_0, ..., a_{pos-1}], and we need to choose a_pos.
            """
            if pos == 0:
                # a_0 must be 0
                return [0]
            
            # a_pos must be > a_{pos-1} and <= m
            # Also, for each i < pos, the difference a_pos - a_i must not be in used_diffs
            # and must be > 0 (which is guaranteed since a_pos > a_i)
            
            # Additionally, the differences among already-chosen elements are already accounted for
            # in used_diffs (they were added when we placed each element).
            
            # Let's also check consistency: the differences between elements in current_row
            # must all be distinct and not in used_diffs (except those already added).
            
            # Actually, we add differences to used_diffs as we place elements.
            # So when we place a_pos, we add {a_pos - a_i for i in range(pos)} to used_diffs.
            
            # Let's collect all differences that would be introduced by placing a candidate value.
            
            prev_val = current_row[-1] if pos > 0 else -1
            
            # We need a_pos > prev_val (if pos > 0) and a_pos <= m
            # And a_pos - a_i not in used_diffs for all i in range(pos)
            
            # Also, we need to ensure that the differences a_pos - a_i are all distinct from each other
            # (they are, since a_i are distinct and a_pos is fixed, so a_pos - a_i are all different)
            # and distinct from already-used differences.
            
            # Let's also consider: can we prune based on the fact that we still need to place
            # (k - pos) more elements in this row, and (n_rows - row_idx - 1) more rows?
            
            # For now, let's just try candidates in order.
            
            candidates = []
            start = prev_val + 1
            for val in range(start, m + 1):
                new_diffs = {val - a for a in current_row}
                if not new_diffs & used_diffs:
                    candidates.append(val)
            
            return candidates
        
        def backtrack(row_idx, pos, current_row):
            """
            Recursive backtracking.
            row_idx: which row we're building (0-indexed)
            pos: which position in the current row (0-indexed)
            current_row: list of values placed so far in this row (length = pos)
            """
            if pos == k + 1:
                # Row is complete
                if row_idx == n_rows - 1:
                    # All rows complete
                    return True
                # Start next row
                return backtrack(row_idx + 1, 0, [])
            
            # Get candidates for current_row[pos]
            candidates = propagate(row_idx, pos, current_row)
            
            for val in candidates:
                new_row = current_row + [val]
                new_diffs = {val - a for a in current_row}
                
                # Save state
                old_used = set(used_diffs)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, pos + 1, new_row):
                    return True
                
                # Restore state
                used_diffs.clear()
                used_diffs.update(old_used)
            
            return False
        
        # Start search
        return backtrack(0, 0, [])
    
    # Try increasing bounds
    # Lower bound: at least total_diffs, but also each row needs k+1 elements
    # A simple lower bound: m >= k (trivial), but we need enough room for distinct differences
    # Let's start from a reasonable value
    lb = max(k, total_diffs)
    
    # For efficiency, let's try a range of bounds
    # The best known for (7,5) is 112, so let's try from 105 to maybe 120
    
    # But we want to minimize, so try from lb upwards
    for m in range(lb, lb + 50):  # Try up to 50 above the lower bound
        start_time = time.time()
        result = search(m)
        if result:
            # We found a solution, but we need to return the actual rows
            # The search above doesn't store the rows, so let's redo with storage
            pass
    
    # The above approach has issues: search() doesn't return the rows.
    # Let's restructure to store the solution.
    
    def search_with_solution(m):
        """
        Backtracking search that stores the solution.
        Returns a list of rows if found, None otherwise.
        """
        used_diffs = set()
        rows = [[] for _ in range(n_rows)]
        
        def backtrack(row_idx, pos, current_row):
            """
            Returns True if a solution is found from this state.
            """
            if pos == k + 1:
                # Row is complete
                if row_idx == n_rows - 1:
                    return True
                return backtrack(row_idx + 1, 0, [])
            
            prev_val = current_row[-1] if pos > 0 else -1
            
            for val in range(prev_val + 1, m + 1):
                new_diffs = {val - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                # Place val
                current_row.append(val)
                rows[row_idx][pos] = val
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, pos + 1, current_row):
                    return True
                
                # Undo
                used_diffs.difference_update(new_diffs)
                current_row.pop()
            
            return False
        
        if backtrack(0, 0, []):
            return rows
        return None
    
    # Try increasing bounds
    lb = max(k, total_diffs)
    
    # For (7,5), lb = 105. Best known is 112.
    # Let's try from lb to lb + 20
    for m in range(lb, lb + 25):
        start_time = time.time()
        rows = search_with_solution(m)
        if rows is not None:
            # Verify the solution
            all_diffs = set()
            valid = True
            for row in rows:
                if len(row) != k + 1:
                    valid = False
                    break
                if row[0] != 0:
                    valid = False
                    break
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            if valid:
                return rows
        if time.time() - start_time > 60:
            # Timeout for this bound, try next
            continue
    
    # If we couldn't find a solution within the time limit, fall back to greedy
    used, rows = set(), []
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
