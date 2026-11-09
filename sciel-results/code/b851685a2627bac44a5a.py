"""Improved program for difference triangle set (4,7) using backtracking with bitsets."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct over all rows.
    Minimizes the largest entry.
    
    Uses backtracking with bitset-based difference tracking for efficiency.
    """
    # Total number of differences per row: C(k+1, 2) = k*(k+1)/2
    # For k=7: 28 differences per row, 4 rows: 112 total differences
    # We need to find 112 distinct positive integers that can be partitioned into 4 groups of 28,
    # each group forming the difference set of a ruler with 8 marks starting at 0.
    
    # Target: minimize the largest entry. Best known is 134.
    # We'll try to find a solution with max entry as small as possible.
    
    # Strategy: 
    # 1. Use backtracking to build rows one by one.
    # 2. Track all used differences in an integer bitset.
    # 3. For each row, try to find k marks (excluding the initial 0) such that
    #    all pairwise differences are distinct and not in used_diffs.
    # 4. Use forward checking: when adding a mark, check if all new differences are unused
    #    and if there are enough remaining differences for the rest of the row.
    
    def try_construct(max_entry):
        """Try to construct n_rows rows with max entry <= max_entry."""
        # Total differences needed: n_rows * k * (k+1) // 2
        total_diffs_needed = n_rows * k * (k + 1) // 2
        
        # We need total_diffs_needed distinct positive differences, each <= max_entry
        # So we need max_entry >= total_diffs_needed (necessary condition)
        if max_entry < total_diffs_needed:
            return None
        
        # Global bitset of used differences across all completed rows
        global_used = 0
        rows = []
        
        def build_row(current_marks, current_diffs_bitset):
            """
            Recursively build a single row.
            current_marks: list of marks in current row (starts with [0])
            current_diffs_bitset: bitset of differences within current_marks
            
            Returns: list of marks if success, None if fail
            """
            if len(current_marks) == k + 1:
                return current_marks
            
            L = len(current_marks)
            rem = k + 1 - L
            
            # Forward check: 
            # We need to add 'rem' more marks.
            # The next mark adds L+1 diffs, next adds L+2, ..., last adds L+rem.
            # Total new diffs needed = sum_{i=1}^{rem} (L + i) = rem*L + rem*(rem+1)//2
            needed_diffs = rem * L + rem * (rem + 1) // 2
            
            # Available diffs in [1, max_entry] not in global_used and not in current_diffs_bitset
            # Note: global_used and current_diffs_bitset are disjoint by construction
            used_mask = global_used | current_diffs_bitset
            # Count available diffs: max_entry - popcount(used_mask & ((1 << (max_entry + 1)) - 1))
            # Since used_mask only has bits up to max_entry set (differences are <= max_entry),
            # we can just do:
            available = max_entry - used_mask.bit_count()
            
            if available < needed_diffs:
                return None
            
            last_mark = current_marks[-1]
            
            # Try next mark from last_mark + 1 to max_entry
            for next_mark in range(last_mark + 1, max_entry + 1):
                # Calculate new differences
                new_diffs = 0
                valid = True
                for m in current_marks:
                    d = next_mark - m
                    # Check if d is already in global_used or current_diffs_bitset
                    bit = 1 << d
                    if (global_used & bit) or (current_diffs_bitset & bit):
                        valid = False
                        break
                    new_diffs |= bit
                
                if not valid:
                    continue
                
                # Recurse
                result = build_row(current_marks + [next_mark], current_diffs_bitset | new_diffs)
                if result is not None:
                    return result
            
            return None
        
        # Build rows one by one
        for _ in range(n_rows):
            row = build_row([0], 0)
            if row is None:
                return None
            
            # Update global_used with the differences of this row
            # We can recompute or pass it up. Recomputing is easy:
            # Differences are all pairs (i, j) with i < j in row
            row_diffs = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    row_diffs |= (1 << d)
            
            # Check for conflicts (should be none if build_row worked correctly with global_used)
            # But build_row only checked against global_used at the time of addition.
            # Since global_used didn't change during build_row, it's safe.
            
            global_used |= row_diffs
            rows.append(row)
        
        return rows
    
    # Try to find a solution with max entry as small as possible
    # Start from the counting bound
    counting_bound = n_rows * k * (k + 1) // 2
    
    # Try values from counting_bound upward
    # The known answer is 134. Counting bound is 112.
    # We try up to 140 to be safe.
    for max_entry in range(counting_bound, counting_bound + 30):
        result = try_construct(max_entry)
        if result is not None:
            return result
    
    # Fallback: Greedy construction (might yield larger max entry)
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            valid = True
            for a in row:
                d = x - a
                bit = 1 << d
                if (used & bit):
                    valid = False
                    break
                new |= bit
            if valid:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
