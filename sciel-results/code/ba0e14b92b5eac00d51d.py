import time
import random
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0, minimizing the
    largest entry, subject to all positive differences within rows being distinct
    across all rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s/150s limit

    def now():
        return time.time()

    def diff_set_of_row(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def validate(rows):
        used = set()
        for r in rows:
            if not r or r[0] != 0:
                return False
            for a, b in zip(r, r[1:]):
                if b <= a:
                    return False
            for d in diff_set_of_row(r):
                if d in used:
                    return False
                used.add(d)
        return True

    def max_val(rows):
        return max(max(r) for r in rows)

    # ---------- Greedy row builder ----------
    def greedy_row(used, temp=0.0, scope_limit=None):
        """
        Build one row greedily. 
        temp: probability to skip a valid small candidate to try larger ones (randomness).
        scope_limit: if set, fail if we exceed this max value.
        """
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope_limit is not None and x > scope_limit:
                return None, used
            
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            
            # Temperature: with probability temp, skip this valid x to explore larger values
            if temp > 0 and random.random() < temp:
                continue
                
            row.append(x)
            used = used | new_diffs
            
        return row, used

    # ---------- Fast backtracking for a single row ----------
    def backtrack_row(used_global, scope_limit, time_limit=None):
        """
        Find a valid row of size k+1 starting at 0, max <= scope_limit,
        such that its differences do not conflict with used_global.
        """
        if time_limit is not None and now() > time_limit:
            return None
        
        row = [0]
        used_local = set(used_global)
        target_len = k + 1
        
        def dfs(row, used_local):
            if time_limit is not None and now() > time_limit:
                return None
            
            if len(row) == target_len:
                return row[:]
            
            last = row[-1]
            rem = target_len - len(row)
            
            # We need to pick 'rem' more numbers.
            # The smallest possible sequence is last+1, last+2, ..., last+rem
            # The largest possible value for the last element is scope_limit.
            # So the current element x must satisfy: x + (rem - 1) <= scope_limit
            max_x = scope_limit - (rem - 1)
            
            if last + 1 > max_x:
                return None
            
            # Try candidates in increasing order
            # Optimization: We can jump ahead if small candidates are likely to fail? 
            # For now, linear scan is okay for k=5,6.
            for x in range(last + 1, max_x + 1):
                if time_limit is not None and now() > time_limit:
                    return None
                
                # Check conflicts with used_local (which includes global used + local row diffs)
                conflict = False
                for a in row:
                    d = x - a
                    if d in used_local:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Pruning: Check if it's even possible to complete the row
                # This is a weak check, but better than nothing.
                # The new diffs are x - a for a in row.
                # We add them to a temp set and recurse.
                
                new_row = row + [x]
                new_used = set(used_local)
                for a in row:
                    new_used.add(x - a)
                
                res = dfs(new_row, new_used)
                if res is not None:
                    return res
            return None

        return dfs(row, used_local)

    # ---------- Initial Construction ----------
    # Try multiple temperatures to find a good starting point
    best_rows = None
    best_max = float('inf')
    
    temps = [0.0, 0.1, 0.2, 0.3, 0.5]
    for temp in temps:
        if now() > deadline - 30:
            break
        used = set()
        rows = []
        ok = True
        for _ in range(n_rows):
            row, used = greedy_row(used, temp=temp)
            if row is None:
                ok = False
                break
            rows.append(row)
        
        if ok:
            mv = max_val(rows)
            if mv < best_max:
                best_max = mv
                best_rows = [r[:] for r in rows]

    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row, used = greedy_row(used, temp=0.0)
            best_rows.append(row)
        best_max = max_val(best_rows)

    # ---------- Iterative Row Replacement ----------
    # Try to reduce the max value by replacing rows one by one.
    # We prioritize replacing the row with the largest max value.
    
    improved = True
    while improved and now() < deadline - 10:
        improved = False
        # Sort rows by their last element (max) descending
        # We want to shrink the largest ones first
        indices = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        
        for i in indices:
            if now() > deadline - 10:
                break
                
            current_max_row = best_rows[i][-1]
            # Target: strictly smaller than current max
            target_scope = current_max_row - 1
            
            # If target_scope is too small, skip? 
            # Theoretical min for k+1 elements is k (0,1,...,k) but diffs collide.
            # For k=5 (6 elements), min scope is around 10-15?
            if target_scope < k: 
                continue

            # Collect diffs from all OTHER rows
            other_diffs = set()
            for j in range(n_rows):
                if j != i:
                    other_diffs |= diff_set_of_row(best_rows[j])
            
            # Try to find a new row for index i with max <= target_scope
            # Use backtracking
            new_row = backtrack_row(other_diffs, target_scope, time_limit=deadline - 10)
            
            if new_row is not None:
                # Validate the whole set just in case (should be valid by construction)
                new_rows = best_rows[:]
                new_rows[i] = new_row
                if validate(new_rows):
                    new_max = max_val(new_rows)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = new_rows
                        improved = True
                        # If we improved, the indices might change, but the loop continues
                        # We can break early if we hit a very good bound? No, keep going.

    # ---------- Final Polish: Local Search / Swaps ----------
    # If we have time, try to swap elements within rows or between rows to tighten further.
    # This is complex, so we stick to the row replacement which is robust.
    # However, we can try to re-greedy the whole thing with the current best_max as a hard cap?
    # No, greedy doesn't respect hard caps well without backtracking.
    
    # One more pass: try to reduce the global best_max by 1, 2, 3...
    # by forcing the max row to be smaller.
    # This is effectively what the loop above does, but sometimes it gets stuck in a local minimum
    # where reducing row A forces row B to expand.
    # The loop handles this by re-evaluating all rows.

    return best_rows

# EVOLVE-BLOCK-END
