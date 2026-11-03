import time
import sys
import random

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # 1. Greedy baseline
    def greedy():
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

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Backtracking with binary search on Scope M
    # Lower bound: We need at least n * (k+1) numbers? No, differences must be distinct.
    # Counting bound: Number of distinct differences is n * k(k+1)/2.
    # Max difference is M. So M(M+1)/2 >= n * k(k+1)/2 => M >= sqrt(n*k(k+1)) roughly.
    # Actually, differences are from 1..M. There are M possible differences.
    # We need N_diff = n * k(k+1)/2 distinct differences.
    # So M >= N_diff.
    # For n=7, k=5: N_diff = 7 * 15 = 105. So M >= 105.
    
    lb = n_rows * k * (k + 1) // 2
    if lb < 1: lb = 1
    
    # We know best_scope is a valid upper bound.
    # We want to find if there is a solution with M < best_scope.
    # We binary search M in [lb, best_scope - 1].
    
    def can_construct(M):
        # Returns True if a solution exists with max entry <= M
        used_diffs = 0
        rows = []
        
        # Bitmask check helper
        # Diffs are 1..M. We use bits 1..M.
        
        def backtrack(row_idx, curr_row, last_val):
            nonlocal used_diffs, rows
            
            if time.time() > deadline:
                raise TimeoutError()
                
            if row_idx == n_rows:
                return True
            
            if len(curr_row) == k + 1:
                # Row complete
                rows.append(list(curr_row))
                # Start next row
                res = backtrack(row_idx + 1, [0], 0)
                if res:
                    return True
                rows.pop()
                return False
            
            # Determine next value x
            # x must be > last_val and <= M
            # Also, the new differences must not conflict with used_diffs
            
            # Optimization: 
            # The smallest possible next value is last_val + 1.
            # The largest is M.
            
            # We can prune if the number of remaining slots in the row 
            # cannot be filled with values <= M given the current last_val.
            # Need (k+1 - len(curr_row)) more values.
            # Max value is M.
            # Min value is last_val + 1.
            # So we need M - last_val >= (k+1 - len(curr_row)).
            if M - last_val < (k + 1 - len(curr_row)):
                return False
            
            for x in range(last_val + 1, M + 1):
                # Calculate new differences
                # new_diffs = {x - a for a in curr_row}
                # Check if any diff is in used_diffs
                valid = True
                new_mask = 0
                for a in curr_row:
                    d = x - a
                    # d is between 1 and M
                    bit = 1 << d
                    if used_diffs & bit:
                        valid = False
                        break
                    new_mask |= bit
                
                if not valid:
                    continue
                
                # Accept x
                curr_row.append(x)
                used_diffs |= new_mask
                
                if backtrack(row_idx, curr_row, x):
                    return True
                
                # Backtrack
                curr_row.pop()
                used_diffs ^= new_mask
                
            return False
        
        # Start with row 0, curr_row [0], last_val 0
        # Note: 0 is not a positive difference, so it doesn't affect used_diffs mask initially.
        # But wait, the first element is 0. The differences are calculated from 0.
        # So for the first element 0, there are no differences.
        # The loop above handles this: curr_row starts as [0].
        # When we pick x, d = x - 0 = x.
        
        return backtrack(0, [0], 0)

    # Binary search
    lo = lb
    hi = best_scope - 1
    found_better = False
    
    # If lb >= best_scope, no need to search
    if lo < hi:
        while lo <= hi:
            mid = (lo + hi) // 2
            if time.time() > deadline - 5: # Reserve time for finalization
                break
            
            # Try to solve for scope mid
            # We need to capture the rows if it succeeds
            # Modify can_construct to return rows
            
            # Let's refactor can_construct to return rows or None
            def solve_for_scope(M):
                used_diffs = 0
                rows = []
                
                def backtrack(row_idx, curr_row, last_val):
                    nonlocal used_diffs, rows
                    if time.time() > deadline:
                        raise TimeoutError()
                    
                    if row_idx == n_rows:
                        return True
                    
                    if len(curr_row) == k + 1:
                        rows.append(list(curr_row))
                        res = backtrack(row_idx + 1, [0], 0)
                        if res:
                            return True
                        rows.pop()
                        return False
                    
                    if M - last_val < (k + 1 - len(curr_row)):
                        return False
                    
                    for x in range(last_val + 1, M + 1):
                        valid = True
                        new_mask = 0
                        for a in curr_row:
                            d = x - a
                            bit = 1 << d
                            if used_diffs & bit:
                                valid = False
                                break
                            new_mask |= bit
                        
                        if not valid:
                            continue
                        
                        curr_row.append(x)
                        used_diffs |= new_mask
                        
                        if backtrack(row_idx, curr_row, x):
                            return True
                        
                        curr_row.pop()
                        used_diffs ^= new_mask
                        
                    return False
                
                try:
                    if backtrack(0, [0], 0):
                        return rows
                except TimeoutError:
                    return None
                return None

            res = solve_for_scope(mid)
            if res:
                # Found a valid solution with scope mid
                best_rows = res
                best_scope = mid
                hi = mid - 1
                found_better = True
            else:
                lo = mid + 1

    # 3. Local Search / Simulated Annealing to improve scope
    # If we have time, try to reduce the scope further using local search.
    # This is a structural change from pure backtracking.
    
    if time.time() < deadline - 10:
        rows = [list(r) for r in best_rows]
        current_scope = best_scope
        
        # Precompute differences for each row
        def get_diffs(row):
            diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.add(row[j] - row[i])
            return diffs
        
        # Global used differences
        all_diffs = set()
        for r in rows:
            all_diffs |= get_diffs(r)
        
        T = 50.0
        cooling = 0.99
        min_T = 0.5
        
        while T > min_T and time.time() < deadline - 2:
            # Pick a random row and a random element (not the first one)
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = random.randint(1, len(row) - 1)
            
            old_val = row[pos]
            
            # Try to move the element to a smaller value
            # to reduce the scope, or just random perturbation.
            # Since we want to minimize max entry, bias towards smaller values.
            # But we must maintain strictly increasing order.
            
            min_allowed = row[pos-1] + 1
            max_allowed = row[pos+1] - 1 if pos < len(row) - 1 else current_scope + 10
            
            if min_allowed > max_allowed:
                continue
            
            # Random new value
            # Bias towards smaller values
            if random.random() < 0.5:
                new_val = random.randint(min_allowed, min(old_val, max_allowed))
            else:
                new_val = random.randint(min_allowed, max_allowed)
            
            if new_val == old_val:
                continue
            
            # Calculate change in differences
            # Remove old differences involving pos
            # Add new differences involving pos
            
            # For simplicity, recompute row diffs and check global uniqueness
            # This is O(k^2) per move, which is fine for k=5.
            
            old_row_diffs = get_diffs(row)
            
            row[pos] = new_val
            new_row_diffs = get_diffs(row)
            
            # Check if any new diff is in (all_diffs - old_row_diffs)
            # i.e., is there a conflict with other rows?
            # Also check if new_row_diffs has duplicates (shouldn't, but just in case)
            # Actually, if the row is strictly increasing, diffs within the row are distinct.
            
            # The set of diffs used by OTHER rows is all_diffs - old_row_diffs
            # But wait, all_diffs is the union of ALL rows.
            # If we remove old_row_diffs, we are left with diffs from other rows.
            # We must ensure new_row_diffs is disjoint from (all_diffs - old_row_diffs).
            
            # Note: all_diffs might have duplicates? No, it's a set.
            # So, conflict if (new_row_diffs & (all_diffs - old_row_diffs)) is not empty.
            
            # However, all_diffs - old_row_diffs might remove diffs that are shared?
            # No, diffs are distinct globally. So no diff is shared.
            # So all_diffs is the disjoint union of diffs from each row.
            
            # So, valid if new_row_diffs is disjoint from (all_diffs - old_row_diffs).
            
            # Let's compute the set of "forbidden" diffs: all_diffs - old_row_diffs
            # But computing this set every time is expensive.
            # Alternative:
            # Check if any d in new_row_diffs is in all_diffs.
            # If d is in all_diffs, it must be from another row (since old_row_diffs are removed from consideration? No).
            # Wait.
            # If d in new_row_diffs and d in all_diffs:
            #   If d is in old_row_diffs, it's fine (it was there before, just moved).
            #   If d is NOT in old_row_diffs, it's a conflict with another row.
            
            # So:
            # conflict = False
            # for d in new_row_diffs:
            #     if d in all_diffs and d not in old_row_diffs:
            #         conflict = True
            #         break
            
            # But "d not in old_row_diffs" is tricky because old_row_diffs is the set BEFORE move.
            # If d is in new_row_diffs but not in old_row_diffs, it's a new diff.
            # If that new diff is in all_diffs (which contains diffs from other rows AND old_row_diffs),
            # then it must be from another row.
            
            # So:
            # for d in new_row_diffs:
            #     if d in all_diffs and d not in old_row_diffs:
            #         conflict = True
            
            # Let's implement this check.
            
            conflict = False
            for d in new_row_diffs:
                if d in all_diffs and d not in old_row_diffs:
                    conflict = True
                    break
            
            if not conflict:
                # Accept move
                # Update all_diffs
                all_diffs = (all_diffs - old_row_diffs) | new_row_diffs
                row[pos] = new_val
                
                new_scope = max(max(r) for r in rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in rows]
                    T = 50.0 # Reset temperature on improvement
                else:
                    # Metropolis criterion
                    delta = new_scope - current_scope
                    if delta < 0 or random.random() < pow(2.71828, -delta / T):
                        # Accept
                        all_diffs = (all_diffs - old_row_diffs) | new_row_diffs
                        row[pos] = new_val
                        current_scope = new_scope
                    else:
                        # Reject
                        row[pos] = old_val
            else:
                # Reject
                row[pos] = old_val
            
            T *= cooling

    return best_rows
