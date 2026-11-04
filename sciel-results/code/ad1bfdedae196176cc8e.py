"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within each row are distinct across all rows.
    Minimizes the largest entry (scope).
    
    Strategy: Binary search on the scope M, using a backtracking solver with bitmask
    optimization for difference tracking.
    """
    import time
    import sys

    def get_greedy_solution(n_rows, k):
        """Generate a valid solution using greedy to establish an upper bound."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                x += 1
            rows.append(row)
        return rows

    def can_construct(M, n_rows, k, node_limit):
        """
        Check if a valid configuration exists with max element <= M.
        Returns the configuration if found, else None.
        Uses backtracking with bitmask for used differences.
        """
        sys.setrecursionlimit(10000)
        used_mask = 0
        rows = []
        nodes_visited = 0
        
        def backtrack(r, c, current_row):
            nonlocal nodes_visited, used_mask
            if nodes_visited > node_limit:
                return False
            
            if r == n_rows:
                return True
            
            if c == k + 1:
                rows.append(current_row)
                if backtrack(r + 1, 0, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Try candidates from last_val + 1 to M
            for cand in range(last_val + 1, M + 1):
                nodes_visited += 1
                if nodes_visited > node_limit:
                    return False
                
                new_diff_mask = 0
                for a in current_row:
                    diff = cand - a
                    new_diff_mask |= (1 << diff)
                
                if new_diff_mask & used_mask:
                    continue
                
                current_row.append(cand)
                # Update used_mask
                # We need to save the old mask to restore on backtrack
                old_mask = used_mask
                used_mask |= new_diff_mask
                
                if backtrack(r, c + 1, current_row):
                    return True
                
                # Undo
                current_row.pop()
                used_mask = old_mask
            
            return False
        
        if backtrack(0, 0, [0]):
            return rows
        return None

    # 1. Determine Lower Bound
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs
    
    # 2. Determine Upper Bound (Greedy Solution)
    greedy_rows = get_greedy_solution(n_rows, k)
    ub = max(max(r) for r in greedy_rows)
    
    # 3. Binary Search for minimal M
    start_time = time.time()
    time_limit = 110
    
    best_rows = greedy_rows # Fallback
    node_limit = 2000000 # 2 million nodes per check
    
    # If the range is small, iterate. Else, binary search.
    # We want to find the smallest M.
    
    if ub - lb < 30:
        for M in range(lb, ub + 1):
            if time.time() - start_time > time_limit:
                break
            res = can_construct(M, n_rows, k, node_limit)
            if res is not None:
                best_rows = res
                break
    else:
        low = lb
        high = ub
        while low < high and (time.time() - start_time) < time_limit:
            mid = (low + high) // 2
            res = can_construct(mid, n_rows, k, node_limit)
            if res is not None:
                best_rows = res
                high = mid
            else:
                low = mid + 1
        
        # If we exited the loop, we might have a best_rows from a previous successful check
        # or the greedy fallback.
        # To be safe, if we didn't find anything better than greedy, keep greedy.
        # But if we found something, it's stored in best_rows.
        # Note: If can_construct fails due to node limit, we treat it as False.
        # This might lead to a suboptimal result if the optimal M was "too hard" to solve
        # within the node limit, but a larger M was easy.
        # However, for this problem, smaller M is usually harder (more constraints).
        # So if M fails, M+1 might also fail or succeed.
        # If we skip the optimal M, we might return a larger M.
        # But greedy is a valid solution, so we always have a valid answer.

    return best_rows
# EVOLVE-BLOCK-END
