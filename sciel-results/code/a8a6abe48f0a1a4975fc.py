import sys
from itertools import combinations

def construct():
    # Increase recursion limit just in case
    sys.setrecursionlimit(2000)
    
    n = 12
    points = range(n)
    
    # Generate all 5-subsets and map them to IDs
    all_5_subsets = list(combinations(points, 5))
    subset_to_id = {s: i for i, s in enumerate(all_5_subsets)}
    num_5_subsets = len(all_5_subsets) # 792
    
    # Generate all 6-subsets and map them to IDs
    all_6_subsets = list(combinations(points, 6))
    num_6_subsets = len(all_6_subsets) # 924
    
    # For each 6-subset, find the IDs of the 5-subsets it contains
    # row_cols[r] = list of column IDs (5-subset IDs) covered by row r (6-subset)
    row_cols = [None] * num_6_subsets
    for r, block in enumerate(all_6_subsets):
        cols = []
        # A 6-subset contains 6 five-subsets
        for i in range(6):
            # Remove the i-th element to get a 5-subset
            # Block is a tuple of 6 elements
            sub = block[:i] + block[i+1:]
            cols.append(subset_to_id[sub])
        row_cols[r] = cols
        
    # column_rows[c] = list of row IDs (6-subset indices) that cover column c (5-subset ID)
    column_rows = [set() for _ in range(num_5_subsets)]
    for r in range(num_6_subsets):
        for c in row_cols[r]:
            column_rows[c].add(r)
    
    # Convert sets to lists for faster iteration if needed, but sets are fine for membership
    # We will use a recursive solver
    
    # Initial available columns: all 792
    initial_available = set(range(num_5_subsets))
    
    solution = []
    
    def solve(available_cols, selected_rows):
        if not available_cols:
            # Found a solution
            solution.extend(selected_rows)
            return True
            
        # Heuristic: Choose the column with the fewest available rows
        # We need to check which rows for each column are still "available"
        # A row is available if all its columns are in available_cols
        
        min_count = float('inf')
        best_col = -1
        
        # To optimize, we can iterate through available_cols
        # But checking all rows for each column might be slow if not careful.
        # However, num_5_subsets is 792.
        # For each col, we have a set of rows.
        # We need to filter rows that are compatible with available_cols.
        
        # Optimization: 
        # Instead of filtering rows for every column, we can just pick a column 
        # and filter its rows.
        
        # Let's iterate through available_cols to find the one with min valid rows
        # This is O(C * R_avg)
        
        for c in available_cols:
            valid_rows = []
            for r in column_rows[c]:
                # Check if row r is compatible with available_cols
                # i.e., all cols in row_cols[r] are in available_cols
                # We can do this by checking intersection or loop
                # Since row size is 6, it's fast.
                if all(cc in available_cols for cc in row_cols[r]):
                    valid_rows.append(r)
                    
            if len(valid_rows) == 0:
                return False # Dead end
                
            if len(valid_rows) < min_count:
                min_count = len(valid_rows)
                best_col = c
                best_rows = valid_rows
                
                # Early termination if count is 1
                if min_count == 1:
                    break
        
        # Try each row in best_rows
        for r in best_rows:
            # Select row r
            # New available cols = available_cols - {cols in row r}
            new_available = available_cols - set(row_cols[r])
            
            if solve(new_available, selected_rows + [r]):
                return True
                
        return False

    # Start solving
    if solve(initial_available, []):
        # Map row indices back to the actual blocks
        blocks = [list(all_6_subsets[r]) for r in solution]
        return blocks
    else:
        # Should not happen for this problem
        return []
