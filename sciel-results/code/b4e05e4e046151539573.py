import time
import sys

def construct(n_rows, k):
    """
    Constructs n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = 110  # Safety margin for 120s limit
    
    # The problem statement specifies 7 rows of 6 integers (k=5).
    # The evaluator passes n_rows and k. We handle general cases but optimize for the known instance.
    
    diffs_per_row = (k + 1) * k // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound for M: We need total_diffs_needed distinct differences in [1, M].
    # So M >= total_diffs_needed.
    # Also, the largest element in a row of size k+1 starting at 0 must be at least k (if 0,1,...,k).
    # But differences go up to M.
    
    def check_solution(rows):
        """Verify the solution is valid."""
        if len(rows) != n_rows:
            return False
        used = set()
        max_val = 0
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                if row[i] < 0:
                    return False
                max_val = max(max_val, row[i])
                if i > 0 and row[i] <= row[i-1]:
                    return False
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def solve(max_limit):
        """
        Backtracking search for a valid configuration with all elements <= max_limit.
        Returns the list of rows if found, else None.
        """
        used_diffs = [False] * (max_limit + 1)
        rows = []
        current_row = [0]
        
        # Precompute available count for pruning? 
        # available = max_limit - 1 (indices 1..max_limit) initially.
        available_count = max_limit # diffs are 1..max_limit
        
        def backtrack(row_idx, pos_in_row):
            nonlocal available_count
            
            if time.time() > deadline:
                return False
            
            # Pruning: Check if enough differences remain
            # Diffs needed for current row: 
            # Current row has pos_in_row elements. 
            # It will have k+1 elements.
            # Diffs generated so far in this row: pos_in_row * (pos_in_row - 1) / 2
            # Diffs to be generated in this row: 
            #   New element x will generate pos_in_row diffs.
            #   Total diffs for full row = (k+1)k/2.
            #   Diffs already "consumed" by current_row (internal): 
            #   Let's just count total diffs needed for remaining rows + remaining part of current row.
            
            # Simpler pruning:
            # Total diffs needed for all rows = total_diffs_needed.
            # Total diffs used so far = total_diffs_needed - (diffs remaining).
            # Diffs remaining = 
            #   For rows > row_idx: n_rows - row_idx - 1 rows * diffs_per_row
            #   For current row: 
            #      Current row length L = pos_in_row.
            #      Final length K = k+1.
            #      Diffs in a row of length L is L(L-1)/2.
            #      Diffs in a row of length K is K(K-1)/2.
            #      Diffs to be added to current row = K(K-1)/2 - L(L-1)/2.
            
            rem_full_rows = n_rows - row_idx - 1
            rem_diffs_full = rem_full_rows * diffs_per_row
            
            L = pos_in_row
            K = k + 1
            rem_diffs_curr = (K * (K - 1) // 2) - (L * (L - 1) // 2)
            
            total_rem_diffs = rem_diffs_full + rem_diffs_curr
            
            if available_count < total_rem_diffs:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos_in_row == k + 1:
                # Row complete
                # Symmetry breaking: current_row must be lexicographically >= previous row
                if rows:
                    prev = rows[-1]
                    if current_row < prev:
                        return False
                
                rows.append(list(current_row))
                success = backtrack(row_idx + 1, 1) # Start next row with [0]
                if success:
                    return True
                rows.pop()
                # Restore used_diffs? 
                # We modified used_diffs when building current_row. 
                # We need to unmark the diffs of current_row.
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_diffs[d] = False
                        available_count += 1
                return False
            
            # Determine start of search for next element
            # If pos_in_row == 1, next element > 0.
            # Else next element > current_row[-1]
            start_val = current_row[-1] + 1
            
            # Upper bound for next element
            # We cannot exceed max_limit.
            # Also, we need to fit the remaining elements.
            # The smallest possible future elements are start_val+1, start_val+2...
            # This is a loose bound, max_limit is tighter.
            
            for x in range(start_val, max_limit + 1):
                if time.time() > deadline:
                    return False
                
                # Check if adding x creates any conflict
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d > max_limit:
                        valid = False
                        break
                    if used_diffs[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Mark diffs
                for d in new_diffs:
                    used_diffs[d] = True
                    available_count -= 1
                
                current_row.append(x)
                
                if backtrack(row_idx, pos_in_row + 1):
                    return True
                
                current_row.pop()
                
                # Unmark diffs
                for d in new_diffs:
                    used_diffs[d] = False
                    available_count += 1
            
            return False
        
        # Initialize for first row
        # Row 0 starts with [0]
        # pos_in_row is 1 (since [0] is already there)
        if backtrack(0, 1):
            return rows
        return None

    # Binary search for the minimum max_limit
    # Lower bound: total_diffs_needed (since we need that many distinct diffs in 1..M)
    # Actually, M must be at least the max element.
    # The max element in a row is at least the max diff in that row.
    # So M >= max diff.
    # Min M is at least total_diffs_needed? No.
    # Example: k=1 (2 elems). Diffs=1. n=2. Total diffs=2. M=2?
    # Row 1: 0, 1 (diff 1). Row 2: 0, 2 (diff 2). Max=2.
    # Total diffs=2. M=2.
    
    # Let's start from a reasonable lower bound.
    # For k=5, n=7, total_diffs=105.
    # Known best is 112.
    # Let's try from 105 upwards.
    
    lower = total_diffs_needed
    upper = 200 # Safe upper bound for this small instance
    
    best_solution = None
    
    # If the instance is large, binary search might be better, but for 7x6, linear scan from lower bound is fine and fast.
    # However, to be safe for other n,k, we do a quick check.
    if n_rows * k > 100:
        # Use binary search
        lo = lower
        hi = upper
        while lo < hi:
            mid = (lo + hi) // 2
            if solve(mid):
                hi = mid
                # We need to store the solution for the best mid
                # But solve() returns rows. We can't easily reuse it if we just want the min M.
                # Let's just find the min M first, then solve again for that M to get the rows.
                # Or store the solution.
                best_solution = solve(mid)
            else:
                lo = mid + 1
        if best_solution is None:
             # Fallback to greedy if search fails or times out? 
             # The task says "invalid result ... scores 0". 
             # We must return a valid result.
             # If search fails, we should return a greedy result which is valid but suboptimal.
             pass
    else:
        # Linear search from lower bound
        for m in range(lower, upper + 1):
            if time.time() > deadline:
                break
            sol = solve(m)
            if sol:
                best_solution = sol
                break
    
    if best_solution is None:
        # Fallback to Greedy (Seed behavior)
        # This ensures we return a valid result, even if suboptimal.
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
        
    return best_solution
