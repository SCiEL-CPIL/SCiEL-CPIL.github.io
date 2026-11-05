import random
import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for safety
    
    # Pre-calculate all possible differences for a given scope
    # A difference d is valid if 1 <= d <= scope
    
    # We will use iterative deepening on the scope (max entry).
    # The counting bound is 105, best known is 112.
    # We start from a reasonable lower bound.
    
    # Lower bound: We need n_rows * k * (k+1) / 2 distinct differences.
    # Max difference in a row is scope.
    # Total distinct differences available in [1, scope] is scope.
    # So scope * scope >= n_rows * k * (k+1) / 2 is a loose bound?
    # Actually, each row uses k*(k+1)/2 differences.
    # Total differences used = n_rows * k*(k+1)/2.
    # These must be distinct integers in [1, scope].
    # So scope >= n_rows * k*(k+1)/2.
    # For n=7, k=5: 7 * 15 = 105.
    
    min_scope = n_rows * k * (k + 1) // 2
    # Also, each row has k+1 elements, so the max element must be at least k (if 0..k).
    # But differences constrain it more.
    
    # We'll try scopes starting from min_scope upwards.
    # To be safe, we might start a bit higher if min_scope is too tight.
    # But iterative deepening is good.
    
    # Heuristic: 
    # When choosing the next element x for a row, we want to minimize the impact on future rows.
    # The "impact" is the set of differences consumed.
    # Smaller differences are more "scarce" because they appear in more potential rows?
    # Actually, all differences are just numbers.
    # However, small differences (like 1, 2) are more likely to conflict?
    # No, the constraint is global uniqueness.
    # A better heuristic: "Least Constraining Value".
    # Choose the value x that leaves the most options for the remaining elements in the current row 
    # and the future rows.
    # Approximation: Choose x that uses differences that are "less useful" for tight packing?
    # Or simply: Sort candidates by the number of *new* differences they introduce?
    # All valid candidates introduce the same number of new differences (k+1 - len(row)).
    
    # Alternative Heuristic:
    # Prefer values that keep the "gap" to the upper bound large?
    # Or prefer values that use larger differences?
    # If we use large differences early, we preserve small differences.
    # Small differences are "cheaper" in terms of value?
    # Let's try: Sort candidates by the maximum difference introduced (descending).
    # This tends to pack small values tightly? No, it uses large jumps.
    
    # Let's use a combined heuristic:
    # 1. Primary: Minimize the maximum difference introduced by this step? (Keep values small)
    # 2. Secondary: Maximize the number of remaining valid candidates for the next step?
    
    # Actually, a very effective heuristic for these "difference sets" is to try to pack the row as tightly as possible 
    # (small values) while avoiding conflicts.
    # So, sort candidates by x (ascending).
    # But this is greedy and might fail.
    # Backtracking handles the failures.
    
    # To speed up, we can add a look-ahead:
    # For each candidate x, check if the *next* element can be placed.
    # This is expensive.
    
    # Let's stick to a simple, effective backtracking with good pruning.
    
    def solve_for_scope(scope):
        """
        Tries to find a solution where all entries are <= scope.
        Returns the solution (list of rows) or None.
        """
        nonlocal start_time, time_limit
        
        # State
        rows = []
        used_diffs = set()
        
        # We will build rows one by one.
        # Within a row, we build elements one by one.
        
        def backtrack(row_idx, current_row):
            nonlocal rows, used_diffs
            
            if time.time() - start_time > time_limit:
                return False
            
            # If we have built all rows, check if we are done
            if row_idx == n_rows:
                return True
            
            # If current row is complete, move to next row
            if len(current_row) == k + 1:
                rows.append(current_row)
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Candidates for next value x
            # x > last_val
            # x <= scope
            # Differences x - a for a in current_row must not be in used_diffs
            
            # Generate candidates
            # To improve heuristic, we can generate all valid candidates and sort them.
            
            candidates = []
            for x in range(last_val + 1, scope + 1):
                new_diffs = []
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    # Heuristic score:
                    # We want to prefer x that is small (to keep scope tight)
                    # But also maybe prefer x that uses "large" differences?
                    # Let's try sorting by x first.
                    # If that fails, we backtrack.
                    candidates.append(x)
            
            if not candidates:
                return False
            
            # Optional: Lookahead / Forward Checking
            # For the best candidate, check if the *next* step has any options?
            # This can be expensive.
            
            # Sort candidates.
            # Strategy: Try small x first (greedy packing).
            # This usually works well for minimizing the max value.
            # If we hit a dead end, we backtrack and try larger x.
            
            for x in candidates:
                # Calculate new diffs
                new_diffs = {x - a for a in current_row}
                
                # Check again (redundant if we checked in generation, but safe)
                # Actually we checked in generation.
                
                # Update state
                old_used = used_diffs
                used_diffs = used_diffs | new_diffs
                
                # Recurse
                if backtrack(row_idx, current_row + [x]):
                    return True
                
                # Restore state
                used_diffs = old_used
            
            return False
        
        # Start backtracking
        if backtrack(0, [0]):
            return [r[:] for r in rows]
        
        return None

    # Iterative deepening
    # Start from min_scope.
    # If no solution, increment scope.
    
    # Optimization: If we fail for a long time, we might be in a bad part of the search space.
    # We can add randomization to the candidate order in later scopes?
    # Or just stick to deterministic for now, as the problem size is small.
    
    # However, pure deterministic backtracking can be slow if the search tree is deep and branching is high.
    # For n=7, k=5, the depth is 7*6=42.
    # Branching factor can be up to scope.
    # This is too large for naive backtracking.
    
    # We need better pruning.
    # Pruning: If the number of remaining required differences > number of available differences <= scope, fail.
    # Remaining rows: r_rem = n_rows - row_idx
    # If current row is incomplete, it will consume (k+1 - len(current_row)) * (k+1 - len(current_row) + 1) / 2? No.
    # Each new element adds len(current_row) new differences.
    # Total differences to be added for current row completion:
    # If we have m elements, we need k+1-m more.
    # The number of differences added by the next element is m.
    # The number of differences added by the next is m+1.
    # ...
    # Total new diffs for current row = sum_{i=m}^{k} i = (k*(k+1) - (m-1)*m) / 2
    
    # Total diffs needed for all remaining rows (including current completion)
    # We can precompute the "cost" in terms of difference count.
    
    # Let's refine the backtracking with this count-based pruning.
    
    def solve_for_scope_pruned(scope):
        nonlocal start_time, time_limit
        
        rows = []
        used_diffs = set()
        
        # Precompute difference counts for partial rows
        # diff_count[m] = number of differences in a row of length m (starting with 0)
        # Actually, we just need the *additional* differences needed.
        
        def backtrack(row_idx, current_row):
            nonlocal rows, used_diffs
            
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            # Pruning: Count remaining differences needed
            # Current row has m = len(current_row) elements.
            # It needs k+1-m more elements.
            # The next element will create m differences.
            # The next will create m+1 differences.
            # ...
            # The last will create k differences.
            # Total new diffs for this row = sum_{i=m}^{k} i
            
            m = len(current_row)
            diffs_needed_current = sum(range(m, k + 1))
            
            # Full rows remaining: n_rows - row_idx - 1
            # Each full row needs k*(k+1)/2 differences.
            diffs_needed_full = (n_rows - row_idx - 1) * k * (k + 1) // 2
            
            total_diffs_needed = diffs_needed_current + diffs_needed_full
            
            # Available differences:
            # Total possible differences in [1, scope] is scope.
            # Used: len(used_diffs)
            # Available: scope - len(used_diffs)
            
            if total_diffs_needed > scope - len(used_diffs):
                return False
            
            last_val = current_row[-1]
            
            # Generate candidates
            candidates = []
            for x in range(last_val + 1, scope + 1):
                new_diffs = []
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    candidates.append((x, new_diffs))
            
            if not candidates:
                return False
            
            # Heuristic: Sort by x (ascending)
            # This encourages tight packing.
            candidates.sort(key=lambda item: item[0])
            
            for x, new_diffs in candidates:
                new_diffs_set = set(new_diffs)
                
                old_used = used_diffs
                used_diffs = used_diffs | new_diffs_set
                
                if backtrack(row_idx, current_row + [x]):
                    return True
                
                used_diffs = old_used
            
            return False
        
        if backtrack(0, [0]):
            return [r[:] for r in rows]
        return None

    # We will try scopes starting from min_scope.
    # To handle cases where min_scope is too tight or search is slow,
    # we will increase scope by 1 each time.
    
    # Also, we can add a time check inside the loop to switch to a larger scope if stuck?
    # No, iterative deepening is best for finding the *minimum*.
    
    # However, if the search for scope S is very slow, we might time out.
    # We can add a node limit or time limit per scope?
    # The global time limit is 115s.
    
    # Let's run the loop.
    current_scope = min_scope
    
    # It's possible that min_scope is not achievable due to structural constraints.
    # We will iterate up to a reasonable upper bound (e.g., 200).
    
    while current_scope <= 200:
        if time.time() - start_time > time_limit:
            break
            
        sol = solve_for_scope_pruned(current_scope)
        if sol is not None:
            return sol
        
        current_scope += 1
        
        # If we are taking too long on a specific scope, we might want to skip to a higher one?
        # But we want the minimum.
        # If we fail at S, we must try S+1.
        
    # If we didn't find a solution within the time or scope limit,
    # we should return a valid solution found earlier or a fallback.
    # But the loop returns as soon as a solution is found.
    # If it times out, we haven't found one.
    
    # Fallback: Greedy construction (might not be optimal but valid)
    # The problem asks for the *best* known, but the score is the value.
    # If we can't find the optimal, we should return the best valid solution we can find quickly.
    # The greedy approach is fast.
    
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
    
    # This greedy solution might have a large scope.
    # To improve the score, we should try to find *any* solution that is better than greedy
    # even if it's not optimal, if we ran out of time on the optimal search.
    # But the iterative deepening stops at the first valid scope.
    # If it times out, it means it didn't finish checking any scope?
    # No, it checks scope by scope.
    # If it times out during scope S, it returns None.
    # Then we return the greedy one.
    
    # However, we can try to run the backtracking for a *larger* scope with a time limit
    # to find a "good" solution if the optimal search fails.
    # But for now, the greedy fallback is safe.
    
    return rows
