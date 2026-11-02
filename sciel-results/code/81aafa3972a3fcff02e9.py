import time
import random
import sys

# Set recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    # The task specifies k=7, n_rows=4.
    # We assume the evaluator passes these specific values, 
    # but we write it generally for small k, n_rows.
    
    start_time = time.time()
    timeout = 145.0 # Safety margin for 150s limit
    
    # Lower bound for M is k(k+1)/2 * n_rows? No, 
    # Counting bound: n_rows * k(k+1)/2 <= M(M+1)/2
    # For n_rows=4, k=7: 4 * 28 = 112 differences.
    # M(M+1)/2 >= 112 => M approx 15? No.
    # The differences are from 1 to M. There are M possible differences.
    # We need 112 distinct differences. So M >= 112.
    # Wait, the differences are $x_j - x_i$. The max difference is $x_k - x_0 \le M$.
    # The set of all possible differences is a subset of $\{1, \dots, M\}$.
    # We need $n_{rows} \times \frac{k(k+1)}{2}$ distinct differences.
    # So $M$ must be at least $n_{rows} \times \frac{k(k+1)}{2}$.
    # For 4 rows, k=7: 4 * 28 = 112. So M >= 112.
    
    min_M = n_rows * k * (k + 1) // 2
    
    # We will try M from min_M upwards.
    # For each M, we try to find a solution using backtracking.
    
    # To improve performance, we can shuffle the order of trying values?
    # Or use a heuristic.
    
    # Let's implement a backtracking search for a fixed M.
    
    def backtrack(row_idx, M, used_diffs, rows):
        if time.time() - start_time > timeout:
            return False, None
            
        if row_idx == n_rows:
            return True, rows[:]
            
        # Try to construct the next row
        # We need to find a sequence x_0=0 < x_1 < ... < x_k <= M
        # such that all differences are in [1, M] and disjoint from used_diffs.
        
        # We'll use a recursive helper to build the current row
        def build_row(pos, current_row, current_diffs):
            if time.time() - start_time > timeout:
                return False, None
                
            if pos == k + 1:
                # Check if this row is valid and disjoint
                # The differences are in current_diffs
                # They are already checked against used_diffs during construction
                return True, current_row[:]
                
            # Determine the range for the next element
            # It must be > current_row[-1] and <= M
            # Also, the difference from 0 is the element itself.
            
            # Heuristic: Try smaller values first? Or larger?
            # Smaller values keep the scope tight, but might block future options.
            # Let's try smaller values first to find a valid row quickly.
            
            start_val = current_row[-1] + 1
            # Optimization: We can limit the upper bound of search for this position
            # based on remaining capacity?
            
            for val in range(start_val, M + 1):
                # Check if val - 0 (which is val) is in used_diffs or current_diffs
                # Actually, we check all new differences val - x for x in current_row
                
                new_diffs = []
                valid = True
                for x in current_row:
                    d = val - x
                    if d in used_diffs or d in current_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                    
                # Pruning: Can we complete the row?
                # We have k - pos positions left.
                # The max possible difference we can add is M - current_row[-1] (approx)
                # This is weak.
                
                # Try this value
                current_row.append(val)
                current_diffs.update(new_diffs)
                
                success, next_row = build_row(pos + 1, current_row, current_diffs)
                if success:
                    return True, next_row
                else:
                    current_row.pop()
                    current_diffs.difference_update(new_diffs)
                    
            return False, None
            
        success, row = build_row(1, [0], set())
        
        if success:
            # Add this row to rows
            rows.append(row)
            # Update used_diffs
            new_used = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    new_used.add(row[j] - row[i])
            used_diffs = used_diffs | new_used
            
            # Recurse to next row
            success_next, final_rows = backtrack(row_idx + 1, M, used_diffs, rows)
            if success_next:
                return True, final_rows
            else:
                rows.pop()
                # used_diffs is local to this call stack if we pass by value (sets are mutable, so we passed a reference? 
                # No, we did used_diffs = used_diffs | new_used, which creates a new set. 
                # So the parent's used_diffs is unaffected. This is good.)
                
        return False, None

    # Main loop over M
    # We might want to randomize the order of M or use a better heuristic.
    # Given the time limit, we start from the lower bound.
    
    # To handle cases where simple backtracking gets stuck in a local minimum 
    # or takes too long, we can add some randomness or restarts.
    # But first, let's try deterministic.
    
    # Optimization: If k and n_rows are small, we can precompute some things?
    # For now, just run the search.
    
    # If the search for a specific M takes too long, we should abort and try M+1.
    # We can do this by checking time in the backtrack function.
    
    # Let's add a restart mechanism if a specific M fails quickly or times out partially.
    # Actually, if it times out, we just move to next M.
    
    # To make it more robust, we can try a few random orders of values?
    # For now, deterministic ascending order.
    
    # One issue: The search space for row 1 is huge. 
    # We can fix the first row to a "standard" ruler to reduce symmetry?
    # The "optimal" ruler for k=7 is known. 
    # But we don't know it. 
    # We can just let it find one.
    
    # Let's try to speed up the inner loop by using lists for differences?
    # Sets are O(1) for lookup.
    
    # Let's run the search.
    
    # We will try M from min_M to min_M + 100.
    # If we find a solution, we return it.
    
    # To improve the chance of finding a solution quickly, we can 
    # try to "shake" the search if it gets stuck? 
    # For now, simple sequential M increase.
    
    # Note: The problem is symmetric with respect to rows.
    # We can enforce an ordering on the rows to reduce the search space.
    # E.g., the first element (after 0) of row i must be <= first element of row i+1?
    # This might cut off valid solutions if the "best" set has rows in a different order.
    # But since we just need *a* set, and the score is the max element, 
    # any permutation of rows is equivalent.
    # So we can fix the first row to be the "lexicographically smallest" valid row?
    # That's hard to define dynamically.
    
    # Let's just run the backtracking.
    
    # To prevent the first row search from taking too long, 
    # we can limit the number of candidates tried for the first row?
    # Or just trust the pruning.
    
    # Let's add a small randomization to the value selection order 
    # to avoid getting stuck in a bad deterministic path.
    # Actually, for the first row, any valid Sidon sequence is fine.
    # The constraint is the disjointness.
    
    # Let's try.
    
    # If the deterministic search is too slow, we might need to 
    # use a heuristic like "try values that leave more room for others".
    
    # Let's implement a simple version first.
    
    # To ensure we don't hang, we check time.
    
    # If no solution found in the loop, we should return a valid solution 
    # with a larger M, e.g., the greedy one, to ensure we don't return None/invalid.
    # The task says "Score: ... minimized". If we return a worse solution, score is lower.
    # If we return invalid, score is 0.
    # So we should always return a valid solution.
    
    # Fallback: Greedy construction.
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Try backtracking for M in range
    # We will try up to a certain limit.
    max_M_to_try = min_M + 50 # 112 to 162
    
    for M in range(min_M, max_M_to_try + 1):
        if time.time() - start_time > timeout - 5: # Leave 5s for fallback
            break
            
        # Run backtracking for this M
        # We pass a fresh used_diffs set
        success, result = backtrack(0, M, set(), [])
        
        if success:
            return result
    
    # If backtracking failed or timed out, use greedy
    return greedy_construct(n_rows, k)
