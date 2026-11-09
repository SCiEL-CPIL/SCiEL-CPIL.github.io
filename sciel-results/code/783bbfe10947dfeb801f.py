import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    timeout = 145  # Leave 5s margin for 150s limit
    
    # k+1 integers, starting at 0.
    # n_rows rows.
    
    # Helper to compute differences for a row as a bitset
    def row_diffs(row):
        diffs = 0
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                diffs |= (1 << d)
        return diffs
    
    def is_valid(rows):
        used = 0
        for r in rows:
            if r[0] != 0:
                return False
            for i in range(len(r) - 1):
                if r[i] >= r[i + 1]:
                    return False
            d = row_diffs(r)
            if d & used:
                return False
            used |= d
        return True
    
    # Backtracking with MRV (Minimum Remaining Values) heuristic
    def backtracking_construct(max_scope, rng, time_limit=2.0):
        """
        Try to construct n_rows rows with all entries <= max_scope.
        Uses MRV: at each step, choose the next row to fill that has the
        fewest valid extensions given the current used differences.
        """
        deadline = start_time + time_limit
        used = 0  # Bitset of used differences
        rows = []
        
        def solve(idx):
            if time.time() > deadline:
                return False
            if idx == n_rows:
                return True
            
            # Try to build row idx
            # We'll build it greedily with backtracking
            row = [0]
            local_used = 0
            
            def build_row(pos, remaining):
                if time.time() > deadline:
                    return False
                if pos == k + 1:
                    # Successfully built the row
                    rows.append(row)
                    used |= local_used
                    if solve(idx + 1):
                        return True
                    # Backtrack
                    used &= ~row_diffs(row)
                    rows.pop()
                    return False
                
                # Try next values
                # MRV: try values that leave the most options for the rest?
                # For now, try smallest first (greedy) with backtracking
                x = row[-1] + 1
                while x <= max_scope:
                    new_diffs = 0
                    valid = True
                    for a in row:
                        d = x - a
                        if (used >> d) & 1 or (local_used >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    
                    if valid:
                        row.append(x)
                        local_used |= new_diffs
                        if build_row(pos + 1, remaining - 1):
                            return True
                        row.pop()
                        local_used &= ~new_diffs
                    x += 1
                return False
            
            if build_row(1, k):
                return True
            return False
        
        return solve(0)
    
    # Get initial greedy solution
    def greedy_construct():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = 0
                    valid = True
                    for a in row:
                        d = x - a
                        if (used >> d) & 1:
                            valid = False
                            break
                        if (new_diffs >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    
                    if valid:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows, max(max(r) for r in rows)
    
    greedy_rows, initial_max = greedy_construct()
    best_rows = [list(r) for r in greedy_rows]
    best_max = initial_max
    
    # Try backtracking with progressively tighter bounds
    # Start from a bound slightly below the greedy solution
    for bound in range(initial_max - 1, max(initial_max - 30, 100), -1):
        if time.time() - start_time > timeout:
            break
        
        # Try multiple random restarts of backtracking
        found = False
        for attempt in range(5):
            if time.time() - start_time > timeout:
                break
            
            rng = random.Random(bound * 1000 + attempt)
            # Shuffle the order of trying values?
            # For now, just run backtracking
            if backtracking_construct(bound, rng, time_limit=1.0):
                found = True
                best_max = bound
                # Extract the rows from the backtracking solution
                # The backtracking function modifies rows and used in place
                # We need to capture the solution
                # Actually, the backtracking function doesn't return the rows properly
                # Let's fix that
                break
        
        if found:
            # We need to get the actual rows from the successful backtracking
            # This is tricky because the backtracking function uses closures
            # Let's refactor to return the rows
            pass
    
    # Better approach: use iterative deepening with a proper backtracking solver
    def solve_with_backtracking(target_scope, rng, time_limit=3.0):
        """
        Try to find a solution with max entry <= target_scope.
        Returns the solution as a list of rows, or None if not found.
        """
        deadline = start_time + time_limit
        used = 0
        rows = []
        
        def solve(idx):
            if time.time() > deadline:
                return False
            if idx == n_rows:
                return True
            
            # Build row idx
            row = [0]
            local_used = 0
            
            def build_row(pos):
                if time.time() > deadline:
                    return False
                if pos == k + 1:
                    rows.append(row[:])
                    used |= local_used
                    if solve(idx + 1):
                        return True
                    used &= ~row_diffs(row)
                    rows.pop()
                    return False
                
                x = row[-1] + 1
                while x <= target_scope:
                    new_diffs = 0
                    valid = True
                    for a in row:
                        d = x - a
                        if (used >> d) & 1 or (local_used >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    
                    if valid:
                        row.append(x)
                        local_used |= new_diffs
                        if build_row(pos + 1):
                            return True
                        row.pop()
                        local_used &= ~new_diffs
                    x += 1
                return False
            
            if build_row(1):
                return True
            return False
        
        if solve(0):
            return rows
        return None
    
    # Try to find a better solution using backtracking with tighter bounds
    # Start from the greedy solution's max and go down
    for bound in range(initial_max - 1, max(initial_max - 50, 100), -1):
        if time.time() - start_time > timeout:
            break
        
        for attempt in range(3):
            if time.time() - start_time > timeout:
                break
            
            rng = random.Random(bound * 1000 + attempt)
            solution = solve_with_backtracking(bound, rng, time_limit=2.0)
            if solution is not None:
                best_rows = solution
                best_max = bound
                break
        
        if best_max == bound:
            # Found a solution at this bound, try to go lower
            continue
    
    # Local search to further improve
    def local_search(initial_rows, steps, rng, T0=1.0):
        cur_rows = [list(r) for r in initial_rows]
        cur_scope = max(max(r) for r in cur_rows)
        cur_sum = sum(sum(r) for r in cur_rows)
        cur_obj = cur_scope * 1000000 + cur_sum
        
        best_rows = [list(r) for r in cur_rows]
        best_obj = cur_obj
        best_scope = cur_scope
        
        T = T0
        cooling = 0.999
        
        for it in range(steps):
            if time.time() - start_time > timeout:
                break
            
            r_idx = rng.randrange(n_rows)
            row = cur_rows[r_idx]
            
            move_type = rng.random()
            new_row = None
            valid_move = False
            
            if move_type < 0.5:
                # Attack max
                if len(row) > 1:
                    prev = row[-2]
                    curr = row[-1]
                    if curr > prev + 1:
                        new_val = rng.randrange(prev + 1, curr)
                        if new_val != curr:
                            new_row = row[:-1] + [new_val]
                            valid_move = True
            elif move_type < 0.8:
                # Random interior mutation
                if len(row) > 2:
                    p = rng.randrange(1, len(row) - 1)
                    lo = row[p - 1] + 1
                    hi = row[p + 1] - 1
                    if lo <= hi:
                        new_val = rng.randrange(lo, hi + 1)
                        if new_val != row[p]:
                            new_row = row[:]
                            new_row[p] = new_val
                            valid_move = True
            else:
                # Row reconstruction
                other_used = 0
                for i in range(n_rows):
                    if i != r_idx:
                        other_used |= row_diffs(cur_rows[i])
                
                new_row = [0]
                local_used = 0
                target_max = cur_scope
                while len(new_row) < k + 1:
                    found = False
                    x = new_row[-1] + 1
                    while x <= target_max:
                        new_diffs = 0
                        valid = True
                        for a in new_row:
                            d = x - a
                            if (other_used >> d) & 1 or (local_used >> d) & 1:
                                valid = False
                                break
                            new_diffs |= (1 << d)
                        
                        if valid:
                            new_row.append(x)
                            local_used |= new_diffs
                            found = True
                            break
                        x += 1
                    
                    if not found:
                        new_row = None
                        break
                
                if new_row:
                    valid_move = True
            
            if not valid_move or new_row is None:
                continue
            
            if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                continue
            
            new_diffs = row_diffs(new_row)
            expected_diffs = k * (k + 1) // 2
            if bin(new_diffs).count('1') != expected_diffs:
                continue
            
            other_used = 0
            for i in range(n_rows):
                if i != r_idx:
                    other_used |= row_diffs(cur_rows[i])
            
            if new_diffs & other_used:
                continue
            
            old_diffs = row_diffs(row)
            cur_rows[r_idx] = new_row
            
            cur_used = 0
            for r in cur_rows:
                cur_used |= row_diffs(r)
            
            cur_scope = max(max(r) for r in cur_rows)
            cur_sum = sum(sum(r) for r in cur_rows)
            cur_obj = cur_scope * 1000000 + cur_sum
            
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_rows = [list(r) for r in cur_rows]
                best_scope = cur_scope
            else:
                # SA acceptance
                delta = cur_obj - best_obj
                if delta < 0 or rng.random() < 2.71828 ** (-delta / max(T, 0.01)):
                    best_obj = cur_obj
                    best_rows = [list(r) for r in cur_rows]
                    best_scope = cur_scope
            
            T *= cooling
        
        return best_rows, best_scope
    
    # Run local search
    rng = random.Random(42)
    steps = 5000
    improved_rows, improved_scope = local_search(best_rows, steps, rng)
    if improved_scope < best_max:
        best_rows = improved_rows
        best_max = improved_scope
    
    for seed in range(1, 10):
        if time.time() - start_time > timeout:
            break
        rng = random.Random(seed)
        improved_rows, improved_scope = local_search(best_rows, steps // 2, rng)
        if improved_scope < best_max:
            best_rows = improved_rows
            best_max = improved_scope
    
    return best_rows
# EVOLVE-BLOCK-END
