"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using Backtracking with Iterative Deepening.
    """
    
    def solve_with_limit(limit):
        """
        Try to construct n_rows rows with max entry <= limit.
        Returns the list of rows if successful, else None.
        """
        used_diffs = set()
        rows = []
        
        def backtrack_row(row_idx, current_row, last_val):
            """
            Recursively build the current_row.
            row_idx: which row we are building (0 to n_rows-1)
            current_row: the list of values in the current row so far
            last_val: the last value added to current_row
            """
            if len(current_row) == k + 1:
                # Row is complete
                rows.append(current_row[:])
                if row_idx + 1 == n_rows:
                    return True
                # Move to next row
                return backtrack_row(row_idx + 1, [0], 0)
            
            # Determine the range for the next value
            # The next value must be > last_val and <= limit
            # Also, we must leave enough room for the remaining k - len(current_row) elements?
            # Actually, since we are just finding *any* valid set within the limit, 
            # we don't strictly need to reserve space unless we want to be smarter.
            # However, if we pick a value too large, we might not fit the rest.
            # But since we are minimizing scope via iterative deepening, 
            # finding *any* solution within the limit is enough.
            
            start_val = last_val + 1
            end_val = limit
            
            # Optimization: We need to add (k + 1) - len(current_row) more elements.
            # The smallest possible sequence from start_val is start_val, start_val+1, ...
            # The largest element in that sequence would be start_val + (remaining - 1).
            # So we can restrict end_val to ensure we can still complete the row?
            # No, the elements don't have to be consecutive.
            # But we must have enough "space" in terms of distinct differences?
            # Simple bound: The remaining elements must be > last_val.
            # If we pick v, the next must be > v.
            
            for v in range(start_val, end_val + 1):
                # Check if v creates a conflict with existing used_diffs or current_row diffs
                valid = True
                new_diffs = []
                for u in current_row:
                    d = v - u
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Also check if any new_diffs conflict with each other?
                # No, d = v - u. Since u are distinct, v-u are distinct.
                # But we must ensure they don't conflict with *each other*? No, they are all positive and distinct because u is strictly increasing.
                
                # Check if new_diffs conflict with each other? 
                # v - u1 != v - u2 if u1 != u2. So no internal conflict in the new set.
                
                # Add to temporary state
                current_row.append(v)
                for d in new_diffs:
                    used_diffs.add(d)
                
                # Recurse
                if backtrack_row(row_idx, current_row, v):
                    return True
                
                # Backtrack
                current_row.pop()
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False
        
        # Start building the first row
        if backtrack_row(0, [0], 0):
            return rows
        return None

    # Lower bound estimate: 
    # Total differences needed: n_rows * k * (k+1) / 2
    # Available differences in [1, limit]: limit
    # So limit >= n_rows * k * (k+1) / 2
    # For n=4, k=8: 4 * 8 * 9 / 2 = 144. 
    # Wait, the counting bound in the prompt is 112? 
    # "simple counting bound is 112". 
    # Let's re-calculate. 
    # Number of pairs in one row of size k+1 is C(k+1, 2) = k(k+1)/2.
    # For k=8, C(9,2) = 36.
    # Total diffs = 4 * 36 = 144.
    # So scope must be at least 144? 
    # The prompt says "best known set has largest entry 134... simple counting bound is 112".
    # There is a discrepancy in my calculation or the prompt's "counting bound" definition.
    # Maybe the counting bound refers to something else? Or maybe I am misinterpreting "distinct over all rows".
    # Yes, "all differences ... are distinct over all rows".
    # So we need 144 distinct positive integers.
    # The smallest 144 positive integers are 1..144.
    # So the max difference must be at least 144.
    # But the max entry is the scope. The max difference is at most scope - 0 = scope.
    # So scope >= 144.
    # Why does the prompt say 134 is best known and 112 is bound?
    # Perhaps the prompt implies that differences don't have to be *all* distinct?
    # "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows"
    # This implies global uniqueness.
    # If global uniqueness is required, scope >= 144.
    # If the best known is 134, then either:
    # 1. My understanding of "distinct" is wrong.
    # 2. The prompt's "134" is for a different k or n?
    # 3. The prompt's "112" bound is wrong or refers to a different metric.
    
    # Let's check the prompt again: "Choose 4 rows of 8 increasing integers... distinct over all rows".
    # "8 increasing integers" -> length 8? Or length k=8 meaning 9 integers?
    # "list of k + 1 integers". So k=8 means 9 integers.
    # Pairs = 9*8/2 = 36.
    # 4 rows * 36 = 144 differences.
    # If they must be distinct, max diff >= 144.
    # If best known is 134, then it's impossible with global distinctness.
    
    # Is it possible that "distinct over all rows" means distinct *within* each row, but not across rows?
    # "all differences ... are distinct over all rows" usually means the set of all differences from all rows is a set (no duplicates).
    # However, if the prompt states 134 is best known, and 112 is bound, there is a contradiction with my 144 lower bound.
    
    # Let's look at the "simple counting bound is 112".
    # Maybe k=7? 8 integers?
    # If k=7, length 8. Pairs = 8*7/2 = 28.
    # 4 * 28 = 112.
    # Ah! If the prompt says "4 rows of 8 increasing integers", that usually means length 8.
    # But the interface says `construct(n_rows, k)` and "Each row is a list of k + 1 integers".
    # If the task description says "8 increasing integers", it might imply k=7?
    # "Choose 4 rows of 8 increasing integers...".
    # If the evaluator passes `k`, and the description says "8 integers", it's ambiguous if k=7 or k=8.
    # However, the interface definition is explicit: `k+1` integers.
    # If the description says "8 integers", and `k+1=8`, then `k=7`.
    # Let's assume the evaluator passes `k=7` if the description says "8 integers".
    # But the prompt also says "Difference triangle set (4,7)".
    # The notation (4,7) usually means n=4, k=7? Or n=4, size=7?
    # In combinatorial design, a (v, k, lambda) design has block size k.
    # Here, "Difference triangle set (4,7)" likely means n=4 rows, and the rows have size 7? Or k=7?
    # If k=7, row size is 8.
    # Then total diffs = 4 * C(8,2) = 4 * 28 = 112.
    # This matches the "simple counting bound is 112".
    # And if the bound is 112, and best known is 134, that makes sense (134 > 112).
    # So, the evaluator will likely pass `k=7`.
    # My code should handle whatever `k` is passed.
    
    # So, lower bound L_min = n_rows * (k * (k+1) // 2)
    # We start iterative deepening from L_min.
    
    L_min = n_rows * (k * (k + 1) // 2)
    
    # To be safe and fast, we might start a bit higher or just start at L_min.
    # Backtracking can be slow. 
    # We will try limits from L_min upwards.
    
    # Optimization: Use a time limit or iteration limit to avoid hanging if L_min is too low and no solution exists for a long time.
    # But for small k, it should be fast.
    
    # Let's add a small randomization to the backtracking to escape local traps?
    # No, backtracking is exact. It will find a solution if one exists for the given limit.
    # The only issue is speed.
    
    # To improve speed, we can order the choices. 
    # Trying smaller values first is good for finding solutions with small scope.
    
    # If no solution is found up to a certain point, we might need to increase the limit.
    # We will loop limit from L_min to L_min + 100 (or some reasonable delta).
    
    for limit in range(L_min, L_min + 200):
        result = solve_with_limit(limit)
        if result is not None:
            return result
            
    # Fallback: If backtracking fails or is too slow (shouldn't happen for small k), 
    # or if we run out of limits, return a greedy solution as a last resort.
    # This ensures we always return a valid result.
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
        
    return greedy_construct()

# EVOLVE-BLOCK-END
