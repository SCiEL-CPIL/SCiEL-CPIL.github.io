import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = 112  # Known counting bound for (4, 7)

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                candidates = []
                x = last + 1
                # Look for a few candidates to allow some randomness
                while x <= 200 and len(candidates) < 20:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Fallback: find next valid
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                
                if random.random() < 0.1 and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                
            rows.append(row)
            
        return rows

    def local_search(rows, time_limit):
        """
        Iterated Local Search.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        def get_all_diffs(rows):
            diffs = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        diffs.add(r[j] - r[i])
            return diffs

        def try_decrease_element(rows, r_idx, c_idx):
            """
            Try to decrease element at rows[r_idx][c_idx].
            Returns new_rows if successful, else None.
            """
            curr_val = rows[r_idx][c_idx]
            prev_v = rows[r_idx][c_idx-1]
            
            all_diffs = get_all_diffs(rows)
            remove_set = set()
            row = rows[r_idx]
            for i in range(c_idx):
                remove_set.add(curr_val - row[i])
            for j in range(c_idx + 1, len(row)):
                remove_set.add(row[j] - curr_val)
            
            fixed_diffs = all_diffs - remove_set
            
            for new_val in range(curr_val - 1, prev_v, -1):
                valid = True
                for i in range(c_idx):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                for j in range(c_idx + 1, len(row)):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_left = [new_val - row[i] for i in range(c_idx)]
                new_right = [row[j] - new_val for j in range(c_idx + 1, len(row))]
                if len(set(new_left + new_right)) != len(new_left + new_right):
                    continue
                
                # Check if this actually reduces scope
                # We only care about global scope reduction
                new_row = row[:]
                new_row[c_idx] = new_val
                new_rows = [r[:] for r in rows]
                new_rows[r_idx] = new_row
                return new_rows

        stuck_counter = 0

        while time.time() < start + time_limit:
            # 1. Targeted Descent: Try to reduce the global maximum
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            
            if max_c_idx == 0:
                break
            
            improved = False
            # Try to decrease the max element
            new_rows = try_decrease_element(best_rows, max_r_idx, max_c_idx)
            if new_rows:
                new_scope = get_scope(new_rows)
                if new_scope < best_scope:
                    best_rows = new_rows
                    best_scope = new_scope
                    improved = True
                    stuck_counter = 0
                else:
                    # Even if scope didn't drop, it might help future moves? 
                    # Usually we only accept if scope drops to avoid cycles.
                    pass
            
            if improved:
                continue

            # 2. Try to decrease other elements
            for r_idx in range(n_rows):
                for c_idx in range(1, row_len):
                    if time.time() > start + time_limit:
                        break
                    if r_idx == max_r_idx and c_idx == max_c_idx:
                        continue
                    new_rows = try_decrease_element(best_rows, r_idx, c_idx)
                    if new_rows:
                        new_scope = get_scope(new_rows)
                        if new_scope < best_scope:
                            best_rows = new_rows
                            best_scope = new_scope
                            improved = True
                            stuck_counter = 0
                            break
                if improved:
                    break
            
            if improved:
                continue

            # 3. Perturbation: Randomly move an element to escape local minima
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, row_len - 1)
            row = best_rows[r_idx]
            curr_val = row[c_idx]
            prev_v = row[c_idx-1]
            next_v = row[c_idx+1] if c_idx < row_len-1 else curr_val + 50
            
            all_diffs = get_all_diffs(best_rows)
            remove_set = set()
            for i in range(c_idx):
                remove_set.add(curr_val - row[i])
            for j in range(c_idx + 1, len(row)):
                remove_set.add(row[j] - curr_val)
            fixed_diffs = all_diffs - remove_set
            
            low = prev_v + 1
            high = next_v - 1 if c_idx < row_len - 1 else curr_val + 50
            
            if low <= high:
                for _ in range(5):
                    cand = random.randint(low, high)
                    if cand == curr_val:
                        continue
                    
                    valid = True
                    for i in range(c_idx):
                        d = cand - row[i]
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    for j in range(c_idx + 1, len(row)):
                        d = row[j] - cand
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    new_left = [cand - row[i] for i in range(c_idx)]
                    new_right = [row[j] - cand for j in range(c_idx + 1, len(row))]
                    if len(set(new_left + new_right)) != len(new_left + new_right):
                        continue
                    
                    # Apply perturbation
                    best_rows[r_idx][c_idx] = cand
                    stuck_counter += 1
                    break

            if stuck_counter > 20:
                # Reconstruct a row? Or just reset stuck counter and continue
                stuck_counter = 0

        return best_rows, best_scope

    def backtrack_solver(scope_limit, seed_rows=None):
        """
        Deterministic backtracking to find a solution with scope <= scope_limit.
        If seed_rows is provided, it tries to use that as a heuristic for ordering or initialization.
        """
        used_diffs = set()
        rows = [[0] for _ in range(n_rows)]
        
        # Heuristic: Order rows by how "hard" they are? 
        # For now, process row 0, then 1, etc.
        # Within a row, process elements in order.
        
        def solve(r_idx, c_idx):
            if time.time() > deadline:
                return False
            
            if r_idx == n_rows:
                return True
            
            if c_idx == row_len:
                return solve(r_idx + 1, 0)
            
            row = rows[r_idx]
            if c_idx == 0:
                row[0] = 0
                return solve(r_idx, c_idx + 1)
            
            prev_val = row[c_idx - 1]
            # Try values from prev_val + 1 up to scope_limit
            # Optimization: If we have a seed, we might try the seed value first?
            # But we want to minimize scope, so we try small values first.
            
            # Heuristic: Try values that don't conflict with existing diffs.
            # We can pre-compute a list of valid candidates?
            
            for x in range(prev_val + 1, scope_limit + 1):
                # Check if x creates any duplicate differences
                valid = True
                new_diffs = []
                for i in range(c_idx):
                    d = x - row[i]
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Check internal consistency of new_diffs (they are distinct by construction if row is strictly increasing)
                # But we must ensure they don't conflict with each other? 
                # No, d = x - row[i]. Since row[i] are distinct, d are distinct.
                
                # Tentatively add
                row[c_idx] = x
                for d in new_diffs:
                    used_diffs.add(d)
                
                if solve(r_idx, c_idx + 1):
                    return True
                
                # Backtrack
                for d in new_diffs:
                    used_diffs.remove(d)
                row[c_idx] = 0 # Reset
            
            return False
        
        return solve(0, 0), rows

    # Phase 1: Greedy Init
    best_rows = greedy_init(seed=42)
    best_scope = get_scope(best_rows)
    
    # Phase 2: Local Search
    if time.time() < deadline - 10:
        ls_time = min(40.0, deadline - time.time() - 10)
        refined_rows, refined_scope = local_search(best_rows, ls_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope
    
    # Phase 3: Backtracking Polish
    # Try to find a solution with scope < best_scope
    # We iterate downwards from best_scope - 1 to lower_bound
    # This is expensive, so we limit the number of attempts or time per attempt.
    
    current_target = best_scope - 1
    attempts = 0
    max_attempts = 20 # Limit number of backtracking calls
    
    while current_target >= lower_bound and time.time() < deadline - 5 and attempts < max_attempts:
        # Run backtracking solver with this target
        # To make it faster, we can use the structure of best_rows to guide the search?
        # Standard backtracking is deterministic.
        
        # Add a small randomization to the backtracking order to escape local traps in the solver itself?
        # The solver is deterministic. We can shuffle the rows?
        # Or just run it.
        
        success, sol_rows = backtrack_solver(current_target)
        attempts += 1
        
        if success:
            if validate(sol_rows):
                best_rows = sol_rows
                best_scope = current_target
                # If we found a solution, we can try to go lower
                current_target = best_scope - 1
            else:
                break
        else:
            # No solution found for this scope
            current_target -= 1
            # If we fail, we might be stuck in a local minimum of the solver's search space?
            # Backtracking is complete, so if it returns False, no solution exists for that scope.
            # So we can safely decrement.
            
    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
