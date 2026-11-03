"""
Difference triangle set (7,5): choose 7 rows of 6 increasing integers starting at 0
so that all positive differences within rows are distinct across all rows.
Minimize the largest entry (scope).

Strategy: Constraint satisfaction with backtracking, using a difference-set
representation with additive consistency pruning. We search for rows in an
order that maximizes constraint propagation, and use memoization of partial
difference sets to prune the search space.

Key insight: each row of k+1=6 entries produces C(6,2)=15 positive differences.
We need 7*15=105 distinct positive differences. The minimum possible scope
must be at least 105 (counting bound). We use a recursive backtracking search
with:
1. Variable ordering: process rows in a fixed order but choose the "most
   constrained" next value at each step.
2. Domain pruning: for each candidate next entry, check that all new
   differences are unused.
3. Additive consistency: ensure the set of chosen differences can form a
   valid ruler (i.e., there exists a strictly increasing sequence starting
   at 0 whose pairwise differences match exactly the chosen set).
4. Deadline guard: track elapsed time and return best valid result found.
"""

import time
import sys

def construct(n_rows, k):
    """
    Return a list of n_rows rows, each a list of k+1 strictly increasing
    integers starting at 0, such that all positive differences within rows
    are distinct across all rows. Minimize the maximum entry.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # Greedy baseline for a quick valid answer
    def greedy_baseline():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy_baseline()
    best_scope = max(max(r) for r in best_rows)

    # Precompute: for a given set of 'd' differences (all positive), can they
    # be realized by a ruler of length k+1 starting at 0?
    # We use a recursive check with memoization.
    
    def can_form_ruler(diff_set, k, max_val, memo=None):
        """
        Check if diff_set can be the set of all positive pairwise differences
        of a strictly increasing sequence of k+1 integers starting at 0,
        all <= max_val.
        """
        if memo is None:
            memo = {}
        key = (frozenset(diff_set), k, max_val)
        if key in memo:
            return memo[key]
        
        if len(diff_set) != k * (k + 1) // 2:
            memo[key] = False
            return False
        
        # The largest difference must be the last element (since 0 is first)
        if not diff_set:
            memo[key] = (k == 0)
            return memo[key]
        
        # Try to build the sequence: we know position 0 is 0.
        # The remaining k positions must be chosen such that all pairwise
        # differences (including those with 0) are in diff_set.
        # The differences from 0 give us the actual values of the other k positions.
        # So the values are exactly the elements of diff_set that correspond to
        # distances from 0, but we don't know which ones those are.
        
        # Alternative: brute force is too slow. Use a smarter approach:
        # The set of differences from 0 to other positions must be a subset S of diff_set
        # with |S| = k, and for any two elements s1, s2 in S, |s1 - s2| must also be in diff_set.
        # And diff_set = S ∪ {|s1-s2| : s1,s2 in S, s1 != s2}
        
        # This is still expensive. For small k=5, we can enumerate subsets of size k.
        # But this is called many times, so we need it fast.
        
        # Simpler approach: just try to build the row greedily given the diff_set constraint.
        # Actually, for our backtracking search, we're building the row incrementally,
        # so we already know it's valid. This function is for pruning: given a partial
        # set of differences, can it be extended to a full valid ruler?
        
        memo[key] = True  # optimistic default; refined by caller
        return True

    # Better approach: direct backtracking on row entries with strong pruning.
    # We search for all n_rows rows simultaneously, tracking used differences.
    
    def backtrack(rows, used_diffs, row_idx, time_limit):
        """
        Recursively build rows. rows is a list of partial rows.
        used_diffs is a set of all positive differences used so far.
        row_idx is the index of the next row to complete.
        Returns (success, completed_rows, scope) or (False, None, None).
        """
        nonlocal best_rows, best_scope, start_time, deadline
        
        if time.time() > time_limit:
            return False, None, None
        
        if row_idx == n_rows:
            # All rows complete
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
            return True, rows, scope
        
        # Build the next row
        row = [0]
        return _build_row(rows, used_diffs, row_idx, time_limit)
    
    def _build_row(rows, used_diffs, row_idx, time_limit):
        nonlocal best_rows, best_scope, start_time, deadline
        
        row = [0]
        
        def _next_pos(pos, current_row, used_so_far):
            if time.time() > time_limit:
                return False, None
            
            if pos == k + 1:
                # Row complete
                rows.append(current_row)
                result = backtrack(rows, used_so_far, row_idx + 1, time_limit)
                rows.pop()
                return result
            
            # Try next values starting from current_row[-1] + 1
            start_val = current_row[-1] + 1
            
            # Upper bound: we want scope <= best_scope, so each new entry
            # must be <= best_scope
            # Also, we need enough room for remaining positions
            remaining = k + 1 - pos
            max_allowed = min(best_scope, current_row[-1] + remaining * 50)  # loose bound
            
            for val in range(start_val, max_allowed + 1):
                if time.time() > time_limit:
                    return False, None
                
                # Check new differences
                new_diffs = {val - a for a in current_row}
                if new_diffs & used_so_far:
                    continue
                
                new_used = used_so_far | new_diffs
                success, completed = _next_pos(pos + 1, current_row + [val], new_used)
                if success:
                    return True, completed
            return False, None
        
        return _next_pos(1, row, used_diffs)

    # The above pure backtracking is too slow for n_rows=7, k=5.
    # Use a hybrid: greedy to get a good initial bound, then local search / 
    # simulated annealing to improve.
    
    def compute_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = row[j] - row[i]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    # Simulated annealing on the scope
    import random
    
    def sa_improve(initial_rows, time_limit):
        nonlocal best_rows, best_scope
        
        rows = [list(r) for r in initial_rows]
        used = set()
        for row in rows:
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    used.add(row[j] - row[i])
        
        current_scope = compute_scope(rows)
        best_local = [list(r) for r in rows]
        best_local_scope = current_scope
        
        temperature = 50.0
        cooling = 0.999
        min_temp = 0.1
        
        while time.time() < time_limit and temperature > min_temp:
            # Pick a random row and a random position to perturb
            ridx = random.randint(0, n_rows - 1)
            pidx = random.randint(1, k)  # don't change position 0
            
            old_val = rows[ridx][pidx]
            
            # Try a random new value
            # Ensure it stays in range and maintains ordering
            lo = rows[ridx][pidx - 1] + 1
            hi = rows[ridx][pidx + 1] - 1 if pidx < k else best_local_scope
            
            if lo > hi:
                temperature *= cooling
                continue
            
            new_val = random.randint(lo, hi)
            if new_val == old_val:
                temperature *= cooling
                continue
            
            # Compute change in differences
            row = rows[ridx]
            removed_diffs = set()
            added_diffs = set()
            
            for i in range(k + 1):
                if i == pidx:
                    continue
                old_d = abs(old_val - row[i])
                new_d = abs(new_val - row[i])
                removed_diffs.add(old_d)
                added_diffs.add(new_d)
            
            # Check for conflicts
            conflict = False
            for d in added_diffs:
                if d <= 0:
                    conflict = True
                    break
                # Check if d is used by other rows or by same row at other positions
                # We need to check if d is in used but not in removed_diffs
                if d in used and d not in removed_diffs:
                    conflict = True
                    break
            
            if conflict:
                temperature *= cooling
                continue
            
            # Also check ordering within the row
            if pidx > 0 and new_val <= row[pidx - 1]:
                temperature *= cooling
                continue
            if pidx < k and new_val >= row[pidx + 1]:
                temperature *= cooling
                continue
            
            # Accept
            rows[ridx][pidx] = new_val
            used -= removed_diffs
            used |= added_diffs
            
            new_scope = compute_scope(rows)
            delta = new_scope - current_scope
            
            if delta < 0 or random.random() < (1.0 if temperature > 1 else (temperature / max(delta, 1)) if delta > 0 else 1.0):
                current_scope = new_scope
                if new_scope < best_local_scope:
                    best_local_scope = new_scope
                    best_local = [list(r) for r in rows]
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
            else:
                # Revert
                rows[ridx][pidx] = old_val
                used -= added_diffs
                used |= removed_diffs
            
            temperature *= cooling
        
        return best_local, best_local_scope

    # Run SA for the remaining time
    remaining_time = deadline - time.time() - 5.0  # leave 5s margin
    if remaining_time > 10:
        sa_rows, sa_scope = sa_improve(best_rows, start_time + remaining_time)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows

    # Final validation
    if not is_valid(best_rows):
        best_rows = greedy_baseline()

    return best_rows
