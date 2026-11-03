import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # 1. Generate a baseline greedy solution
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
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Try to find a better solution using backtracking
    # The lower bound is roughly (n_rows * k(k+1)/2)^(1/2) ? No, counting bound is n_rows * k(k+1)/2 distinct diffs.
    # Max diff is M. Number of possible diffs is M(M+1)/2.
    # n_rows * k(k+1)/2 <= M(M+1)/2 => M >= sqrt(n_rows * k(k+1)).
    # For n_rows=7, k=5: 7 * 15 = 105. sqrt(105) ~ 10.2? No.
    # The counting bound given is 105.
    
    lower_bound = 105
    if best_scope <= lower_bound:
        return best_rows
        
    # We will try M from lower_bound to best_scope - 1
    # To be efficient, we might want to try random restarts or different orderings if simple backtracking is slow.
    # But simple backtracking with pruning should be okay for small M.
    
    def search(M, row_idx, used_diffs, current_rows):
        if time.time() > deadline:
            return None
            
        if row_idx == n_rows:
            return current_rows[:]
            
        # Construct next row
        # We need to find 5 intervals d1..d5 such that sum <= M
        # and all 15 diffs are new.
        
        def build_row(idx, current_pos, row_marks, row_diffs):
            if time.time() > deadline:
                return None
            
            if idx == k:
                # Row complete
                # Check if we can proceed to next row
                new_used = used_diffs | row_diffs
                result = search(M, row_idx + 1, new_used, current_rows + [row_marks])
                return result
            
            # Choose next mark
            # Current mark is current_pos
            # Next mark must be > current_pos
            # Max next mark is M - (k - idx) (roughly, to leave room for remaining intervals >= 1)
            # Actually, we just need sum of remaining intervals >= (k - idx).
            # So next_mark <= M - (k - idx).
            
            max_next = M - (k - idx)
            if max_next <= current_pos:
                return None
                
            # Heuristic: try smaller increments first? Or larger?
            # Smaller increments keep scope small, which is good for feasibility.
            # But they create smaller differences, which might conflict more easily?
            # Let's try increasing order.
            
            for next_pos in range(current_pos + 1, max_next + 1):
                # Calculate new differences
                new_diffs = {next_pos - m for m in row_marks}
                
                # Check conflicts with global used
                if new_diffs & used_diffs:
                    continue
                # Check conflicts with current row diffs
                if new_diffs & row_diffs:
                    continue
                # New diffs are distinct by construction (strictly increasing marks)
                
                # Recurse
                new_row_diffs = row_diffs | new_diffs
                res = build_row(idx + 1, next_pos, row_marks + [next_pos], new_row_diffs)
                if res is not None:
                    return res
                    
            return None
            
        # Start building row
        return build_row(1, 0, [0], set())

    # Try scopes from lower bound up to best_scope - 1
    for M in range(lower_bound, best_scope):
        if time.time() > deadline:
            break
        # To avoid getting stuck in one specific search path, we could add some randomization or restarts, 
        # but deterministic backtracking is a good start.
        # We might want to try different orderings of rows? 
        # For now, assume row order doesn't matter for existence, but might for search speed.
        # We can try to fix the first row to be "canonical" or something?
        # Let's just run it.
        
        result = search(M, 0, set(), [])
        if result:
            best_rows = result
            best_scope = M
            # We found a solution for M. Can we go lower?
            # The loop continues to lower_bound, so it will check M-1 next? 
            # Wait, the loop is `for M in range(lower_bound, best_scope)`.
            # If we find a solution for M=110, best_scope becomes 110.
            # The range was calculated before. We should probably restart the loop or just continue.
            # Since we are iterating upwards, if we find M=110, we haven't checked 105..109 yet?
            # No, range(lower_bound, best_scope) iterates 105, 106...
            # So if we find 108, we return it? No, we want the MINIMUM.
            # So we should iterate from lower_bound upwards and return the FIRST one found.
            # If we find one at M=107, that is the best we can get in this loop (assuming 105, 106 failed).
            # So we can return immediately.
            return result
            
    return best_rows
