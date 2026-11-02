"""Difference triangle set construction using backtracking with MRV and time management."""
import time
import sys
import numpy as np

# Deadline: 140s to leave margin before 150s kill
_DEADLINE = 140.0

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within each row are distinct across all rows.
    Minimize the largest entry."""
    start_time = time.time()
    
    # Total number of differences needed: n_rows * k(k+1)/2
    # Each difference must be unique and positive
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Simple lower bound on max entry: we need total_diffs distinct positive integers
    # The smallest possible set of total_diffs distinct positive integers is 1..total_diffs
    # So max entry >= total_diffs (since differences go up to max_entry)
    # Actually, differences can be at most max_entry, so we need at least total_diffs values in [1, max_entry]
    # Thus max_entry >= total_diffs
    lb = total_diffs
    
    # Also, each row has k+1 entries starting at 0, so the largest entry in row i is at least k
    # (if entries are 0,1,2,...,k)
    
    # Try to find a solution with max entry = lb, lb+1, lb+2, ...
    # Use backtracking with MRV
    
    best_solution = None
    best_max = None
    
    # Greedy initial solution as fallback
    greedy_solution = _greedy_construct(n_rows, k)
    if greedy_solution is not None:
        best_solution = greedy_solution
        best_max = max(max(r) for r in greedy_solution)
    
    # Try to improve with backtracking
    for target_max in range(lb, best_max + 1):
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        solution = _backtrack_construct(n_rows, k, target_max, start_time)
        if solution is not None:
            best_solution = solution
            best_max = target_max
            break  # Found optimal for this target_max
    
    if best_solution is None:
        # Should not happen, but safety
        best_solution = _greedy_construct(n_rows, k)
    
    return best_solution


def _greedy_construct(n_rows, k):
    """Greedy construction: place smallest possible next mark."""
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


def _backtrack_construct(n_rows, k, max_entry, start_time):
    """Backtracking search with MRV to construct rows with max entry <= max_entry."""
    # We need to place k+1 entries in each row, starting at 0
    # Each row: a_0=0 < a_1 < ... < a_k <= max_entry
    # All differences a_j - a_i (i<j) must be distinct across all rows
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    # If max_entry < total_diffs, impossible (need that many distinct positive diffs)
    if max_entry < total_diffs:
        return None
    
    # State: 
    # - rows: list of lists, each row being a list of placed entries
    # - used_diffs: set of all differences used so far
    # - current_row_idx: which row we're currently building
    # - current_pos: position in current row we're trying to fill
    
    # MRV approach: for each row, we need to choose k entries from {1, ..., max_entry}
    # such that all pairwise differences are distinct and not in used_diffs
    
    # We'll build rows one at a time, and within a row, choose entries in order
    
    rows = [[0]]  # Start first row with 0
    used_diffs = set()
    
    def backtrack(row_idx, pos_in_row):
        """Try to fill row_idx, position pos_in_row (0-indexed, 0 is already 0)."""
        if time.time() - start_time > _DEADLINE - 5:
            return False
        
        if row_idx == n_rows:
            return True
        
        if pos_in_row == k + 1:
            # Row complete, move to next row
            rows.append([0])
            return backtrack(row_idx + 1, 1)
        
        current_row = rows[row_idx]
        last_val = current_row[-1]
        
        # Try all possible values for current_row[pos_in_row]
        # Must be > last_val and <= max_entry
        # Also, all new differences must not be in used_diffs
        
        # MRV: try values that create the most constrained situation first?
        # Actually, for finding any solution, let's try values in order but with pruning
        
        # Pre-compute: for a candidate value v, the new differences are {v - a for a in current_row}
        # These must all be in [1, max_entry] and not in used_diffs
        
        # To speed up, we can skip values that would create a difference already in used_diffs
        # or a difference > max_entry (but v <= max_entry and a >= 0, so v-a <= max_entry always)
        
        start_val = last_val + 1
        end_val = max_entry
        
        # Try values in increasing order
        for v in range(start_val, end_val + 1):
            if time.time() - start_time > _DEADLINE - 5:
                return False
            
            # Check if v creates any difference already in used_diffs
            valid = True
            new_diffs = set()
            for a in current_row:
                d = v - a
                if d <= 0:
                    valid = False
                    break
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Also check that new_diffs don't have duplicates (they can't since a are distinct)
            # Place v
            current_row.append(v)
            used_diffs |= new_diffs
            
            if backtrack(row_idx, pos_in_row + 1):
                return True
            
            # Undo
            current_row.pop()
            used_diffs -= new_diffs
        
        return False
    
    if backtrack(0, 1):
        # Remove the extra [0] that was added for the next row
        if len(rows) == n_rows + 1:
            rows.pop()
        return rows
    else:
        return None


# Test with small cases
if __name__ == "__main__":
    # Test with n_rows=2, k=2
    result = construct(2, 2)
    print("n_rows=2, k=2:", result)
    for row in result:
        print(row)
    
    # Test with n_rows=3, k=3
    result = construct(3, 3)
    print("n_rows=3, k=3:", result)
    for row in result:
        print(row)
