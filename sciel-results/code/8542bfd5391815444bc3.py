import sys
import time
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Lower bound
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    lb = total_diffs_needed
    
    # We will search for the minimum M such that a valid configuration exists
    # with max entry <= M.
    
    # Bitmask helper
    # We represent used differences as an integer bitmask.
    # Bit i is set if difference i is used.
    
    def solve_for_M(M, timeout=2.0):
        """
        Try to find n_rows valid rows with max entry <= M.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        # State:
        # rows_found: list of completed rows
        # current_row: list of elements in the row currently being constructed
        # used_mask: integer bitmask of used differences
        
        rows_found = []
        
        def backtrack(row_idx, current_row, used_mask):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                # Row complete
                if row_idx == n_rows - 1:
                    rows_found.append(current_row[:])
                    return True
                else:
                    # Start next row
                    rows_found.append(current_row[:])
                    if backtrack(row_idx + 1, [0], used_mask):
                        return True
                    rows_found.pop()
                    return False
            
            # Determine next element
            last_val = current_row[-1]
            # We need to pick x such that last_val < x <= M
            # Also, we must ensure enough room for remaining elements
            # remaining_elements = k - len(current_row) + 1
            # Min x = last_val + 1
            # Max x = M - (k - len(current_row)) # to allow at least 1 step for each remaining
            
            min_x = last_val + 1
            max_x = M - (k - len(current_row))
            
            if min_x > max_x:
                return False
                
            # Optimization: Check if enough differences are available
            # This is a bit complex with bitmask, so we skip for now unless simple
            # For now, just iterate candidates
            
            # To improve search, we can try smaller x first (tighter packing)
            # or larger x? Smaller x tends to keep the row "tighter" which might
            # leave more large differences for other rows?
            # Usually, greedy tight packing is a good heuristic for the first row.
            
            for x in range(min_x, max_x + 1):
                # Calculate new differences
                new_diffs_mask = 0
                conflict = False
                for y in current_row:
                    d = x - y
                    if d > M: # Should not happen if x<=M and y>=0
                        conflict = True
                        break
                    bit = 1 << d
                    if used_mask & bit:
                        conflict = True
                        break
                    new_diffs_mask |= bit
                
                if conflict:
                    continue
                
                # Check if we have enough *potential* differences left?
                # Total available bits in [1, M] is M.
                # Used bits: used_mask.bit_count()
                # New bits: new_diffs_mask.bit_count()
                # Remaining rows need: (n_rows - row_idx - 1) * diffs_per_row
                # If (M - (used_mask.bit_count() + new_diffs_mask.bit_count())) < needed:
                #    continue
                # This is a valid prune!
                used_count = used_mask.bit_count()
                new_count = new_diffs_mask.bit_count()
                remaining_rows = n_rows - row_idx - 1
                needed_diffs = remaining_rows * diffs_per_row
                available = M - (used_count + new_count)
                if available < needed_diffs:
                    continue
                
                # Recurse
                current_row.append(x)
                if backtrack(row_idx, current_row, used_mask | new_diffs_mask):
                    return True
                current_row.pop()
                
            return False
        
        # Start with first row [0]
        if backtrack(0, [0], 0):
            return rows_found
        return None

    best_rows = None
    # Search M from lb upwards
    # We need to be careful with the upper bound.
    # Greedy gives an upper bound.
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
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    # If greedy is small, use it as upper bound. Otherwise, cap it.
    upper_bound = greedy_max
    if upper_bound > 200: # Safety cap for search time
        upper_bound = 200
        
    # Iterate M
    for M in range(lb, upper_bound + 1):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 2:
            break
            
        # Allocate time for this M
        # If M is close to lb, give it more time?
        # Or just give a fixed small time slice?
        # If a solution exists at M, it should be found fast.
        # If not, we waste time.
        # Let's give 1 second per M, but reduce if we are running out of time.
        time_slice = 1.0
        if elapsed > time_limit - 10:
            time_slice = 0.2
            
        result = solve_for_M(M, timeout=time_slice)
        if result is not None:
            best_rows = result
            break
            
    if best_rows is None:
        # Fallback to greedy if no solution found within time/bounds
        best_rows = greedy_rows
        
    return best_rows
