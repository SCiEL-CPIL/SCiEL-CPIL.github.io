import time
import random
import sys
from collections import defaultdict

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows, each of length k+1.
    Strategy: Backtracking Search with Forward Checking and Heuristic Variable Ordering.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Problem parameters
    # Each row has k+1 elements.
    # Element 0 is always 0.
    # Elements 1..k are to be determined.
    # Total elements per row: k+1.
    # Differences per row: (k+1)*k/2.
    # Total differences: n_rows * (k+1)*k/2.
    
    # For n_rows=4, k=7:
    # Diffs per row: 8*7/2 = 28.
    # Total diffs: 4 * 28 = 112.
    
    # The "scope" is the maximum value in any row.
    # We want to minimize the scope.
    # We can frame this as: Find the smallest M such that there exists a valid configuration with max element <= M.
    # However, binary search on M might be slow if we restart from scratch.
    # Instead, we use a direct backtracking search that tries to keep values small,
    # guided by a heuristic that prefers smaller values but allows backtracking.
    
    # State representation:
    # rows: list of lists. rows[i][j] is the j-th element of the i-th row.
    # rows[i][0] = 0 always.
    # used_diffs: set of all differences currently present in the partially constructed rows.
    
    # We will fill rows one by one? Or element by element?
    # Filling row by row is better for "Forward Checking" of the next row's feasibility?
    # Actually, filling element by element across all rows might be too complex for state management.
    # Let's try filling row by row.
    # When constructing row i, we have used_diffs from rows 0..i-1.
    # We need to find a sequence of k integers x_1 < x_2 < ... < x_k such that:
    # 1. All differences x_j - x_i (including 0) are distinct within the row.
    # 2. All differences x_j - x_i are NOT in used_diffs.
    
    # This reduces the problem to finding k rows, one after another, where each row is a 
    # "Sidon-like" sequence (distinct pairwise differences) avoiding a global set of forbidden differences.
    
    # Heuristic for Row Construction:
    # Given forbidden diffs F, find the smallest possible max element for a row of length k+1 (starting with 0)
    # such that all internal diffs are distinct and not in F.
    
    # We can use a backtracking search for each row, or a greedy approach with limited backtracking.
    # Given the small k (7), the search space for a single row is manageable if we prune well.
    
    def find_row(forbidden_diffs, max_limit):
        """
        Finds a row of length k+1 starting with 0, strictly increasing, 
        with distinct internal differences, none of which are in forbidden_diffs.
        Returns the row if found, else None.
        Uses backtracking with a time limit per call to avoid getting stuck.
        """
        row = [0]
        # We need to choose k more elements.
        # Let's use a recursive backtracking function.
        
        # Optimization: Precompute which diffs are forbidden.
        # The search space for the first element x_1 is small.
        # The search space grows, but we can prune if the minimum possible remaining diffs exceed max_limit.
        
        def backtrack(idx, current_max, current_diffs):
            """
            idx: next index to fill (1-based, since 0 is filled)
            current_max: last element in row
            current_diffs: set of differences in the current row so far
            """
            if idx == k + 1:
                return True
            
            # Lower bound for next element: current_max + 1
            # Upper bound: max_limit
            # Also, we must ensure that we can fit (k - idx) more elements after this one.
            # The smallest possible next elements would be current_max+1, current_max+2, ...
            # So we need max_limit - x >= (k - idx)
            
            # Pruning: 
            # If we pick x, the new diffs are x - a for a in row.
            # These must not be in forbidden_diffs or current_diffs.
            
            # Heuristic: Try smallest valid x first to keep scope small.
            # But sometimes a slightly larger x allows a more compact subsequent sequence?
            # Usually, for Sidon sets, smaller is better for packing.
            
            # Estimate minimum additional span required.
            # If we pick x, the remaining k-idx elements must be > x.
            # The minimal span for m elements is m (if consecutive).
            # So we need x + (k - idx) <= max_limit.
            
            min_next = current_max + 1
            max_next = max_limit - (k - idx)
            
            if min_next > max_next:
                return False
            
            # Iterate through possible values for x
            # To speed up, we can skip values that immediately cause a conflict.
            # Also, we can randomize the order slightly if we want to explore different shapes,
            # but for minimizing scope, ascending is best.
            
            # Optimization: Instead of iterating 1 by 1, we can jump.
            # But k is small (7), so 1 by 1 is fine if pruning is good.
            
            for x in range(min_next, max_next + 1):
                # Check conflicts
                valid = True
                new_diffs_to_add = set()
                for a in row:
                    d = x - a
                    if d in forbidden_diffs or d in current_diffs:
                        valid = False
                        break
                    new_diffs_to_add.add(d)
                
                if not valid:
                    continue
                
                # Tentatively add x
                row.append(x)
                current_diffs.update(new_diffs_to_add)
                
                if backtrack(idx + 1, x, current_diffs):
                    return True
                
                # Undo
                row.pop()
                current_diffs.difference_update(new_diffs_to_add)
            
            return False
        
        # Start backtracking
        # We need to be careful with recursion depth and time.
        # k=7, depth 7. Very shallow.
        # The branching factor is the main cost.
        
        # To avoid getting stuck in a bad local minimum for the first row, 
        # we might want to try multiple "max_limit" values or restarts.
        # However, the global search (row by row) handles the interaction.
        
        # Let's add a node counter to prevent excessive time on a single row if max_limit is too tight.
        node_count = 0
        limit_nodes = 5000 # Arbitrary limit per row construction attempt
        
        def backtrack_limited(idx, current_max, current_diffs):
            nonlocal node_count
            node_count += 1
            if node_count > limit_nodes:
                return False
            
            if idx == k + 1:
                return True
            
            min_next = current_max + 1
            max_next = max_limit - (k - idx)
            
            if min_next > max_next:
                return False
            
            for x in range(min_next, max_next + 1):
                valid = True
                new_diffs_to_add = set()
                for a in row:
                    d = x - a
                    if d in forbidden_diffs or d in current_diffs:
                        valid = False
                        break
                    new_diffs_to_add.add(d)
                
                if not valid:
                    continue
                
                row.append(x)
                current_diffs.update(new_diffs_to_add)
                
                if backtrack_limited(idx + 1, x, current_diffs):
                    return True
                
                row.pop()
                current_diffs.difference_update(new_diffs_to_add)
            
            return False
        
        # We call this with an initial max_limit. 
        # If it fails, the caller will increase max_limit.
        # But in our main loop, we don't know the right max_limit for a specific row in isolation.
        # The global scope is what matters.
        
        # Alternative Strategy:
        # Instead of finding a row for a fixed max_limit, we let the backtracking search 
        # naturally find the smallest possible row given the constraints.
        # The "max_limit" in the above function is a hard cap. 
        # If we set max_limit very high, it finds *a* row, not necessarily the *smallest* one.
        # To find the smallest one, we should binary search on max_limit for this row?
        # Or just let the ascending loop in `for x` find the smallest valid x at each step?
        # Yes, `for x in range(...)` tries smallest first. So if we set max_limit to a sufficiently large number,
        # the first solution found by this DFS will be the lexicographically smallest (and thus minimal max element) 
        # valid row for that specific forbidden set.
        
        # So, we just need to set max_limit to something "safe" like 200 or 300.
        # If no solution is found within the node limit, we return None.
        
        row = [0]
        current_diffs = set()
        
        # Use a generous max_limit for the row search itself, 
        # because the global scope constraint is applied at the end or via restarts.
        # Actually, if we are minimizing global scope, we should constrain each row's max to <= current_global_scope.
        # But we don't know the final global scope.
        
        # Let's try a different approach:
        # The main loop will manage the global scope M.
        # We binary search on M.
        # For a given M, can we construct n_rows rows with max element <= M?
        
        return None # Placeholder, logic moved to main loop

    def can_construct_with_scope(M):
        """
        Attempts to construct n_rows rows with all elements <= M.
        Returns True and fills 'best_rows' if successful.
        """
        # Reset state
        used_diffs = set()
        current_rows = []
        
        # We construct rows one by one.
        # Order of rows doesn't matter for validity, but might matter for search efficiency.
        # Usually, constructing the "hardest" row first is better? 
        # Or just 0 to n_rows-1.
        
        for r_idx in range(n_rows):
            # Find a row with max element <= M, avoiding used_diffs.
            row_found = None
            
            # We use a backtracking search to find the row.
            # To ensure we find the *smallest* possible row (to leave more room for others?),
            # we iterate x from small to large.
            
            row = [0]
            current_diffs = set()
            
            # We need a backtracking function that respects M.
            
            def bt_row(idx, curr_max, curr_diffs):
                if idx == k + 1:
                    return True
                
                min_next = curr_max + 1
                # Max next is M, but we also need space for remaining elements.
                # Remaining elements: k - idx
                # They must be strictly greater than x.
                # So x + (k - idx) <= M  => x <= M - (k - idx)
                max_next = M - (k - idx)
                
                if min_next > max_next:
                    return False
                
                # Heuristic: Try values in increasing order.
                # To add diversity if we restart, we could shuffle, 
                # but for a deterministic "can we do it" check, ascending is fine.
                # However, if we get stuck, we might want to try a different structure.
                # For now, simple ascending.
                
                for x in range(min_next, max_next + 1):
                    valid = True
                    new_diffs = set()
                    for a in row:
                        d = x - a
                        if d in used_diffs or d in curr_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    row.append(x)
                    curr_diffs.update(new_diffs)
                    
                    if bt_row(idx + 1, x, curr_diffs):
                        return True
                    
                    row.pop()
                    curr_diffs.difference_update(new_diffs)
                
                return False
            
            # We need to be careful with recursion limit or time.
            # k=7 is small.
            # But if M is small, the search space is small.
            # If M is large, the search space is huge, but we stop at the first solution.
            # The first solution found with ascending x will be the "tightest" row.
            
            # Time check
            if time.time() > deadline - 5.0:
                return False
            
            if bt_row(1, 0, set()):
                # Row found
                current_rows.append(row[:])
                # Update used_diffs
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        used_diffs.add(row[j] - row[i])
            else:
                return False
        
        return True

    # Binary Search on Scope M
    # Lower bound: Counting bound? 
    # Total diffs = n_rows * k * (k+1) / 2.
    # Max diff is M.
    # Number of possible diffs is M.
    # So M >= n_rows * k * (k+1) / 2.
    # For 4,7: 4 * 28 = 112.
    
    low = n_rows * k * (k + 1) // 2
    # Upper bound: Greedy solution scope?
    # Let's find a greedy solution first to get an upper bound.
    
    def greedy_solution():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            curr_max = 0
            while len(row) < k + 1:
                x = curr_max + 1
                while True:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        curr_max = x
                        break
                    x += 1
            rows.append(row)
        return max(max(r) for r in rows)
    
    high = greedy_solution()
    # Add a small buffer just in case greedy is not optimal but close
    high += 10 
    
    best_M = high
    best_rows = None
    
    # We will try to improve M.
    # Since the search space for "can_construct_with_scope" is exponential in the worst case,
    # we should be careful.
    # However, for k=7, the depth is only 7.
    # The branching factor is bounded by M.
    # If M is around 112-134, the branching factor is manageable.
    
    # To make it robust, we can use a randomized backtracking if we get stuck.
    # But let's try deterministic first.
    
    # We'll iterate M from low to high? Or Binary Search?
    # Binary search is risky if the feasibility is not monotonic (it is: if M works, M+1 works).
    # So Binary Search is valid.
    
    # However, the cost of a failed search at a small M might be high if it explores a lot of dead ends.
    # A failed search at a small M is usually fast (pruned quickly).
    # A successful search at a large M is fast (finds solution quickly).
    # The "sweet spot" M where it's hard to find might be expensive.
    
    # Let's do a linear search from low upwards, but with a time limit per M.
    # If a specific M takes too long, we skip to a higher M? 
    # No, if M fails, M+1 might succeed.
    
    # Better: Binary Search, but with a "hard timeout" for each check.
    # If check(M) times out, we assume it's too hard and try a higher M? 
    # Actually, if it times out, we don't know if it's feasible.
    # But usually, if it's infeasible, it prunes quickly. If it's feasible, it finds it quickly.
    # The hard cases are near the boundary.
    
    # Let's just do a few random restarts with the backtracking solver for a fixed M,
    # or use the backtracking solver which is deterministic.
    
    # Since k is small, let's just try M from low to high, incrementing by 1,
    # but limit the search time for each M to, say, 5 seconds.
    # If it exceeds 5s, we assume it's too complex for this M and try M+1?
    # This is not rigorous.
    
    # Alternative: Use the backtracking solver with a node limit.
    # If node limit is exceeded, return False (infeasible or too hard).
    # This makes the check non-deterministic in terms of correctness (false negatives).
    # But for optimization, finding *a* solution is enough.
    
    # Let's refine `can_construct_with_scope` to use a node limit.
    
    def can_construct_with_scope_limited(M, node_limit=100000):
        used_diffs = set()
        current_rows = []
        
        for r_idx in range(n_rows):
            row = [0]
            current_diffs = set()
            node_count = 0
            
            def bt_row(idx, curr_max, curr_diffs):
                nonlocal node_count
                node_count += 1
                if node_count > node_limit:
                    return False
                
                if idx == k + 1:
                    return True
                
                min_next = curr_max + 1
                max_next = M - (k - idx)
                
                if min_next > max_next:
                    return False
                
                # Heuristic: Order values by "promisingness"?
                # Smaller values are generally better for scope minimization.
                # But to avoid getting stuck in a local trap, we could randomize slightly?
                # For now, keep ascending.
                
                for x in range(min_next, max_next + 1):
                    valid = True
                    new_diffs = set()
                    for a in row:
                        d = x - a
                        if d in used_diffs or d in curr_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    row.append(x)
                    curr_diffs.update(new_diffs)
                    
                    if bt_row(idx + 1, x, curr_diffs):
                        return True
                    
                    row.pop()
                    curr_diffs.difference_update(new_diffs)
                
                return False
            
            if bt_row(1, 0, set()):
                current_rows.append(row[:])
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        used_diffs.add(row[j] - row[i])
            else:
                return False, None
        
        return True, current_rows

    # Search for minimal M
    # Start from low bound
    # To speed up, we can jump.
    # But let's try a few values.
    
    # Since the best known is 134, and bound is 112.
    # We will try M from 112 to 134.
    # If we find a solution, we return it.
    # If time runs out, we return the best found (which might be the greedy one).
    
    # Initialize best with greedy
    greedy_rows = None
    used_diffs_greedy = set()
    for _ in range(n_rows):
        row = [0]
        curr_max = 0
        while len(row) < k + 1:
            x = curr_max + 1
            while True:
                valid = True
                for a in row:
                    if (x - a) in used_diffs_greedy:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    used_diffs_greedy.update(x - a for a in row[:-1])
                    curr_max = x
                    break
                x += 1
        if greedy_rows is None:
            greedy_rows = []
        greedy_rows.append(row)
    
    best_scope = max(max(r) for r in greedy_rows)
    best_solution = greedy_rows
    
    # Try to improve
    # We'll iterate M from low to best_scope - 1
    # To avoid spending too much time on infeasible small M, we use the node limit.
    
    # We can also randomize the order of rows or values to escape local minima?
    # The backtracking is deterministic.
    # If it fails, it fails.
    # To add diversity, we could shuffle the order in which we check x values?
    # But we want the smallest max element, so ascending is correct.
    
    # Let's just run the search.
    
    for M in range(low, best_scope):
        if time.time() > deadline - 2.0:
            break
        
        found, rows = can_construct_with_scope_limited(M, node_limit=50000)
        if found:
            best_scope = M
            best_solution = rows
            # Since we are searching upwards, the first M that works is the minimal M 
            # for this specific search strategy.
            # We can break here if we assume the search is complete for this M.
            # But wait, if node_limit was hit, it might have failed incorrectly.
            # If it succeeded, it's a valid solution.
            # We can break because we are looking for the minimum M.
            break
            
    return best_solution
