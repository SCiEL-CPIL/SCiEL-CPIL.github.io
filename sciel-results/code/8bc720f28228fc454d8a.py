import time
import sys

def construct():
    """
    Construct the Witt design S(5,6,12) using Algorithm X (DLX) for Exact Cover.
    The problem is to cover all 5-subsets of 12 points exactly once using 6-subsets.
    There are C(12,5) = 792 columns (5-subsets).
    There are C(12,6) = 924 rows (6-subsets).
    Each row covers 6 columns.
    We need to find a set of rows that covers all columns exactly once.
    The solution size is known to be 132.
    """
    
    # 1. Map 5-subsets to column indices
    from itertools import combinations
    
    points = range(12)
    all_5subsets = list(combinations(points, 5))
    col_index = {s: i for i, s in enumerate(all_5subsets)}
    num_cols = len(all_5subsets) # 792
    
    # 2. Build the matrix: rows are 6-subsets, entries are the indices of 5-subsets they cover
    # We don't need the full matrix, just the list of columns for each row.
    all_6subsets = list(combinations(points, 6))
    rows = []
    for b in all_6subsets:
        cols = [col_index[s] for s in combinations(b, 5)]
        rows.append(cols)
        
    # 3. Implement Algorithm X (DLX)
    # Since the matrix is sparse and small, we can use a recursive backtracking with bitsets or sets.
    # Given the small size, a simple recursive search with "minimum remaining values" heuristic will work.
    
    # Precompute for each column, which rows cover it.
    # col_to_rows[c] = list of row indices that cover column c
    col_to_rows = [[] for _ in range(num_cols)]
    for r_idx, cols in enumerate(rows):
        for c in cols:
            col_to_rows[c].append(r_idx)
            
    # We will use a recursive search.
    # State: 
    #   covered_cols: set of columns already covered
    #   used_rows: set of rows already used (to ensure distinct blocks, though exact cover implies distinct rows if no two rows are identical, which is true here)
    #   current_solution: list of row indices
    
    # Optimization: Use bitmasks for columns? 792 bits is too big for int in Python comfortably, but we can use Python's arbitrary precision integers as bitsets.
    # Let's try bitsets for speed.
    
    # col_mask[c] = 1 << c
    # row_mask[r] = sum of col_mask[c] for c in rows[r]
    
    # Actually, Python sets are fast enough for 792 elements if we use the "most constrained variable" heuristic.
    
    # Let's use a simpler heuristic: pick the column with the fewest available rows.
    
    solution = []
    used_rows = set()
    covered = set()
    
    # To avoid recursion depth issues (though 132 is small), we use recursion.
    # But Python recursion limit might be tight? 132 is fine.
    
    # We need a fast way to check if a row is still valid (no conflict).
    # A row is valid if none of its columns are in 'covered'.
    
    def search(depth):
        if len(covered) == num_cols:
            return True
            
        # Heuristic: Find the column with the minimum number of candidate rows.
        # Candidate rows for a column c are rows that cover c and are not in used_rows.
        # Since we only add rows that don't conflict, 'used_rows' is not strictly necessary for conflict checking if we check 'covered'.
        # But we must ensure we don't pick the same row twice.
        
        min_count = float('inf')
        best_col = -1
        best_candidates = []
        
        # Iterate over uncovered columns
        # To speed this up, we can maintain a list of uncovered columns.
        # But finding the min count requires checking all uncovered columns.
        
        for c in range(num_cols):
            if c in covered:
                continue
                
            # Count available rows for this column
            # A row r is available if r not in used_rows and no column in rows[r] is in covered (other than c? No, if any other column is covered, it's a conflict).
            # Actually, if we pick a row, it must cover c.
            # And it must not cover any other column already in 'covered'.
            
            candidates = []
            for r in col_to_rows[c]:
                if r in used_rows:
                    continue
                # Check if row r conflicts with current covered set
                conflict = False
                for cc in rows[r]:
                    if cc in covered and cc != c: # c is not covered yet, so if cc is covered, it's a conflict
                        conflict = True
                        break
                if not conflict:
                    candidates.append(r)
            
            if len(candidates) < min_count:
                min_count = len(candidates)
                best_col = c
                best_candidates = candidates
                
                if min_count == 0:
                    return False
                if min_count == 1:
                    break # Can't get better than 1
            
        if best_col == -1:
            return False # No uncovered columns? Should be handled by base case.
            
        # Try each candidate row for best_col
        for r in best_candidates:
            # Add row r
            used_rows.add(r)
            new_covered_cols = [cc for cc in rows[r] if cc not in covered]
            for cc in new_covered_cols:
                covered.add(cc)
            
            solution.append(r)
            
            if search(depth + 1):
                return True
            
            # Backtrack
            solution.pop()
            used_rows.remove(r)
            for cc in new_covered_cols:
                covered.remove(cc)
                
        return False

    # Start search
    # Increase recursion limit just in case
    sys.setrecursionlimit(10000)
    
    if search(0):
        # Convert row indices to blocks
        result = []
        for r_idx in solution:
            result.append(list(all_6subsets[r_idx]))
        return result
    else:
        # Fallback: return empty or a partial solution if search fails (should not happen)
        return []

# EVOLVE-BLOCK-START
# The function construct is defined above.
# EVOLVE-BLOCK-END
