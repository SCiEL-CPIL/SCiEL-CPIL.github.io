"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct over all rows.
    Minimize the largest entry (scope).
    
    Uses backtracking with forward checking and constraint propagation.
    """
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on scope
    lb = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    lb = max(lb, k)
    
    def solve(scope):
        """Try to construct n_rows rows with max entry <= scope."""
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row):
            """
            row_idx: which row we're constructing
            current_row: list of entries in current row (starts with [0])
            """
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Row complete, move to next
                rows.append(current_row[:])
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            min_next = current_row[-1] + 1 if current_row else 1
            
            # We need 'rem' more elements after this one
            rem = k + 1 - len(current_row) - 1
            # The largest possible value for x is scope - rem
            # because we need 'rem' more values larger than x
            max_next = scope - rem
            
            if min_next > max_next:
                return False
            
            # Try each possible next value
            for x in range(min_next, max_next + 1):
                new_diffs = []
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Forward checking:
                # 1. Global counting bound
                updated_used_size = len(used_diffs) + len(new_diffs)
                
                # Diffs needed to complete current row
                r_after = len(current_row) + 1
                current_row_remaining_diffs = sum(range(r_after, r_after + rem))
                
                # Diffs needed for future rows
                future_rows_diffs = (n_rows - row_idx - 1) * diffs_per_row
                
                total_remaining_diffs = current_row_remaining_diffs + future_rows_diffs
                
                # Available diffs in [1, scope]
                available = scope * (scope + 1) // 2 - updated_used_size
                
                if available < total_remaining_diffs:
                    continue
                
                # 2. Local feasibility: Is there a valid next value?
                if rem > 0:
                    found_next = False
                    # Next value y must be in [x+1, scope - (rem - 1)]
                    next_min = x + 1
                    next_max = scope - (rem - 1)
                    
                    # To speed up, we can break early if we find one
                    for y in range(next_min, next_max + 1):
                        ok = True
                        # Check against used_diffs and new_diffs
                        # We can't just check new_diffs because y-a might conflict with used_diffs
                        # We need to check y-a against used_diffs U new_diffs
                        for a in current_row:
                            d = y - a
                            if d in used_diffs:
                                ok = False
                                break
                        if ok:
                            for nd in new_diffs:
                                # Check if y-a equals any new diff?
                                # No, y-a is a difference from y to a.
                                # new_diffs are differences from x to a.
                                # They are distinct sets of values generally.
                                # But we must ensure y-a is not in new_diffs?
                                # Actually, new_diffs are already "used" in the context of this branch.
                                # So yes, check against new_diffs too.
                                if y - a == nd:
                                    # This check is redundant if we just check if (y-a) is in used_diffs U new_diffs
                                    # But used_diffs doesn't have new_diffs yet.
                                    # So we check both.
                                    ok = False
                                    break
                        if ok:
                            # Also need to ensure differences between y and x?
                            # d = y - x.
                            # This is not in new_diffs (which are x-a).
                            # Is it in used_diffs?
                            # Yes, we must check y-x against used_diffs.
                            # My loop above only checked y-a for a in current_row.
                            # It did NOT check y-x.
                            # So the check above is INCOMPLETE.
                            # We need to check y - b for all b in current_row + [x].
                            # Let's restart the local feasibility check properly.
                            pass
                    
                    # Proper local feasibility check
                    found_next = False
                    temp_used = used_diffs.union(new_diffs)
                    next_min = x + 1
                    next_max = scope - (rem - 1)
                    for y in range(next_min, next_max + 1):
                        ok = True
                        for b in current_row + [x]:
                            d = y - b
                            if d in temp_used:
                                ok = False
                                break
                        if ok:
                            found_next = True
                            break
                    if not found_next:
                        continue
                
                # Place x
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, current_row):
                    return True
                
                # Undo
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        return backtrack(0, [0])
    
    start_time = time.time()
    timeout = 110
    
    # Start from a slightly higher bound if we suspect the counting bound is weak
    # But let's stick to lb first.
    for scope in range(lb, lb + 150):
        if time.time() - start_time > timeout:
            break
        
        result = solve(scope)
        if result:
            return rows
    
    # Fallback: greedy
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

# EVOLVE-BLOCK-END
