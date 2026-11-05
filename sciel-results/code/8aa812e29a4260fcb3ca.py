import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for 150s limit
    
    # n_rows=7, k=5 (so 6 entries per row: 0 + 5 others)
    # Total differences per row = 5 * 6 / 2 = 15
    
    def get_diff_mask(row):
        """
        Returns a bitmask of all positive differences in the row.
        row is a list of integers [0, a1, ..., ak]
        """
        mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    def backtrack(depth, used_mask, min_row_tuple, M):
        """
        Finds 'n_rows - depth' rows that are disjoint from 'used_mask',
        lexicographically greater than 'min_row_tuple',
        with max element <= M.
        Returns a list of rows or None.
        """
        if depth == n_rows:
            return []
        
        # Time check
        if time.time() - start_time > time_limit:
            return None

        # We need to generate a row R > min_row_tuple
        # R = [0, x1, x2, x3, x4, x5]
        # x1 > 0, x1 < x2 < ... < x5 <= M
        
        # To enforce R > min_row_tuple, we can iterate x1, x2... 
        # with lower bounds derived from min_row_tuple.
        
        # Let min_row = list(min_row_tuple)
        # If min_row is empty, lower bounds are 1, 2, ...
        # Actually, lexicographic order:
        # R > min_row iff at the first index i where R[i] != min_row[i], R[i] > min_row[i].
        
        # We can construct R incrementally.
        # State: current partial row, current index in row, current mask for this row.
        
        # To keep it simple and fast, we will iterate through valid rows using a recursive generator 
        # that respects the lower bound.
        
        # However, generating ALL valid rows > min_row is expensive.
        # We will use a DFS to build the row, pruning conflicts with used_mask.
        
        row = [0]
        row_mask = 0
        
        # Lower bounds for x_i
        # If min_row_tuple is empty, lb = [1, 2, 3, 4, 5] (since strictly increasing)
        # If min_row_tuple is (a1, a2, a3, a4, a5), 
        # we can pick x1 >= a1. If x1 > a1, rest are free (>= x1+1).
        # If x1 == a1, x2 >= a2. If x2 > a2, rest free.
        # etc.
        
        # We implement this by trying values for x1, then x2, etc.
        
        def dfs_row(pos, last_val, current_mask):
            """
            Builds row[pos+1] ... row[k]
            pos: current index in row (0-based), so we are choosing row[pos+1]
            last_val: row[pos]
            current_mask: mask of diffs generated so far in this row
            """
            if time.time() - start_time > time_limit:
                return None
            
            if pos == k: # row has k+1 elements (0 to k)
                # Check if this row is strictly greater than min_row_tuple
                # This check is handled by the search space definition below
                # Verify no internal conflicts (already done by construction)
                # Verify disjoint from used_mask
                if current_mask & used_mask:
                    return None
                
                # Success for this row
                sub_rows = backtrack(depth + 1, used_mask | current_mask, tuple(row), M)
                if sub_rows is not None:
                    return [row[:]] + sub_rows
                return None
            
            # Determine lower bound for next element
            # We are choosing row[pos+1]
            # Let min_next = 0
            # If min_row_tuple is not empty:
            #   If for all j <= pos, row[j] == min_row_tuple[j], then row[pos+1] >= min_row_tuple[pos+1]
            #   Else, row[pos+1] > row[pos] (standard strictly increasing)
            
            # To enforce R > min_row_tuple:
            # We can just generate all R > last_val, and check R > min_row_tuple at the end?
            # No, that's inefficient.
            
            # Let's determine the effective lower bound `lb` for `x`.
            lb = last_val + 1
            
            # Check if we are still "equal" to min_row_tuple so far
            equal_so_far = True
            if min_row_tuple:
                for i in range(pos + 1):
                    if row[i] != min_row_tuple[i]:
                        equal_so_far = False
                        break
            
            if equal_so_far and min_row_tuple:
                # We must pick x > min_row_tuple[pos] if we want to be strictly greater?
                # No, if we pick x == min_row_tuple[pos], we continue matching.
                # If we pick x > min_row_tuple[pos], we are strictly greater.
                # If we pick x < min_row_tuple[pos], we are smaller (invalid).
                # So if equal_so_far, x >= min_row_tuple[pos] is allowed, 
                # BUT if we pick x == min_row_tuple[pos], we must ensure the rest of the row 
                # makes the whole row > min_row_tuple.
                # Actually, if we pick x == min_row_tuple[pos], we are NOT yet strictly greater.
                # We will be strictly greater only if a later element is larger.
                # If we end up with R == min_row_tuple, it is NOT strictly greater.
                
                # So, if equal_so_far:
                #   We can pick x in [min_row_tuple[pos], M]
                #   BUT if we pick x == min_row_tuple[pos], we must ensure the suffix is > min_row_tuple suffix.
                #   This is complex.
                
                # Simpler approach:
                # Iterate x from lb to M.
                # If x < min_row_tuple[pos] (and equal_so_far), skip.
                # If x == min_row_tuple[pos] (and equal_so_far), we are still equal.
                # If x > min_row_tuple[pos] (and equal_so_far), we are now strictly greater.
                
                if x < min_row_tuple[pos]: # pseudo-code
                    continue
                
            # Actually, let's just pass a flag `is_greater` or handle it in the loop.
            # For now, let's assume we generate x > last_val.
            # And we check `tuple(row) > min_row_tuple` at the end of row construction?
            # That might generate many invalid rows.
            
            # Let's use the lower bound logic properly.
            # If not equal_so_far: lb = last_val + 1
            # If equal_so_far: lb = min_row_tuple[pos]  (if pos < len(min_row_tuple))
            
            # Wait, if min_row_tuple is (1, 2, 3, 4, 5) and we are at pos=0 (choosing x1).
            # row[0]=0. min_row_tuple[0]=1.
            # If we pick x1=1, equal_so_far remains True.
            # If we pick x1=2, equal_so_far becomes False (2 > 1).
            # If we pick x1=0, invalid (must be > 0).
            
            # So:
            if min_row_tuple and equal_so_far:
                # We can pick x >= min_row_tuple[pos]
                # But x must be > last_val.
                lb = max(lb, min_row_tuple[pos])
            
            # Now iterate x from lb to M
            for x in range(lb, M + 1):
                # Check if x creates a conflict within the row
                new_diffs = []
                conflict = False
                for a in row:
                    d = x - a
                    if d <= 0: continue
                    if (current_mask >> d) & 1:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if conflict:
                    continue
                
                # Check if x creates a conflict with used_mask
                # We can check this incrementally or at the end.
                # Incremental is better for pruning.
                new_mask = current_mask
                for d in new_diffs:
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                    new_mask |= (1 << d)
                
                if conflict:
                    continue
                
                # Recurse
                row.append(x)
                res = dfs_row(pos + 1, x, new_mask)
                if res is not None:
                    return res
                row.pop()
                
            return None
        
        return dfs_row(0, 0, 0)

    # Main loop
    best_rows = None
    # Counting bound for (7, 5) is 105.
    # Known best is 112.
    # We start from 105.
    for M in range(105, 120):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Try to find a solution with max element <= M
        # To avoid finding the same solution multiple times (permutations),
        # we enforce lexicographic ordering of rows.
        # The first row can be any valid row.
        # We start with min_row = () (empty), so any row is allowed.
        # But to reduce symmetry, we can fix the first row to be the smallest possible?
        # No, just letting the backtracking handle it with min_row=() is fine,
        # but it might explore many branches.
        # We can add a heuristic: try rows with smaller max element first?
        # The DFS naturally tries smaller x first.
        
        # One optimization: 
        # If we find a solution, we return it.
        
        res = backtrack(0, 0, (), M)
        
        if res is not None:
            # Verify
            # Check validity
            valid = True
            all_diffs = set()
            for r in res:
                if len(r) != k + 1:
                    valid = False
                    break
                if r[0] != 0:
                    valid = False
                    break
                for i in range(1, len(r)):
                    if r[i] <= r[i-1]:
                        valid = False
                        break
                mask = get_diff_mask(r)
                if mask & all_diffs:
                    valid = False
                    break
                all_diffs.update([d for d in range(1, M+1) if (mask >> d) & 1])
            
            if valid:
                best_rows = res
                break # Found minimal M
        
    if best_rows is None:
        # Fallback: Greedy construction to ensure we return *something* valid
        # This might have a larger scope, but it's better than crashing.
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
