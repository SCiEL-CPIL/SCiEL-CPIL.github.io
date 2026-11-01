import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    def get_diffs(row):
        # Returns set of all positive differences in the row
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

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

    def find_best_row(k, forbidden_diffs):
        """
        Find a row of k+1 elements starting with 0, strictly increasing,
        such that its differences are disjoint from forbidden_diffs.
        Minimizes the maximum element (last element).
        Returns the row or None.
        """
        # Lower bound for max element M
        # Number of differences is N = k(k+1)/2
        # They must be distinct integers from 1 to M.
        # So M >= N.
        N = k * (k + 1) // 2
        lb = N
        
        # Upper bound
        # A safe upper bound. Greedy usually produces rows with max < 200 for small k.
        # Let's use 500.
        ub = 500

        def backtrack(row, forbidden, max_allowed):
            if len(row) == k + 1:
                return True
            
            last = row[-1]
            rem = k + 1 - len(row)
            
            # We need to pick 'rem' more numbers, all > last and <= max_allowed.
            # The smallest possible sequence is last+1, last+2, ..., last+rem.
            # So we need last + rem <= max_allowed.
            # This is a necessary condition.
            
            # Iterate next values x from last+1 to max_allowed
            # Optimization: If x - last is too small, we might not be able to fit the rest?
            # No, the constraint is just strict increase.
            
            # Heuristic: Try values that minimize the new differences?
            # Or just try in increasing order to find small M solutions first?
            # Since we are fixing M in the outer loop, we just need *a* solution.
            # To find a solution with small max element, we iterate M in outer loop.
            # Inside, we just need to find *any* valid completion.
            
            # To speed up backtracking, we can check conflicts quickly.
            
            for x in range(last + 1, max_allowed + 1):
                # Pruning: Can we fit the remaining rem-1 numbers?
                if max_allowed - x < rem - 1:
                    break
                
                new_diffs = {x - a for a in row}
                if new_diffs & forbidden:
                    continue
                
                row.append(x)
                if backtrack(row, forbidden, max_allowed):
                    return True
                row.pop()
            
            return False

        # Try M from lb to ub
        for M in range(lb, ub + 1):
            row = [0]
            if backtrack(row, forbidden_diffs, M):
                return row
        
        return None

    # 1. Initial Solution
    rows = greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)

    # 2. Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Precompute differences for all rows
        all_diffs = [get_diffs(r) for r in rows]
        
        for i in range(n_rows):
            if time.time() > deadline:
                break
            
            # Calculate forbidden differences (all diffs except row i)
            forbidden = set()
            for j in range(n_rows):
                if j != i:
                    forbidden.update(all_diffs[j])
            
            # Find the best possible row for position i
            new_row = find_best_row(k, forbidden)
            
            if new_row is not None:
                old_max = max(rows[i])
                new_max = max(new_row)
                
                # Accept if strictly better
                if new_max < old_max:
                    rows[i] = new_row
                    all_diffs[i] = get_diffs(new_row)
                    improved = True
                    
                    # Update global best
                    current_max = max(max(r) for r in rows)
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [r[:] for r in rows]
    
    return best_rows
