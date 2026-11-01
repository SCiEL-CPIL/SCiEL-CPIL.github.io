import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid approach: 
    1. Greedy/SA to find a good initial upper bound.
    2. Binary/Linear search on the scope (max entry) using a Backtracking 
       feasibility check that selects rows one by one, prioritizing rows with 
       minimal max entry to maximize space for subsequent rows.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def _validate(rows, n_rows, k):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _global_max(rows):
        return max(max(row) for row in rows)

    # --- Phase 1: Initial Solution via SA/Greedy ---
    
    def local_search_init(n_rows, k, seed=None):
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def simulated_annealing(rows, n_rows, k, deadline):
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        all_used = set()
        row_diffs = []
        for r in current_rows:
            d = _row_diffs(r)
            all_used |= d
            row_diffs.append(d)
        
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        
        T = 500.0
        T_min = 0.5
        cool = 0.99998
        no_improve = 0
        max_no_improve = 20000
        last_reopt_time = time.time()
        reopt_interval = 1.0
        
        def _reoptimize_row(row, other_diffs, k, max_scope=None):
            n = k + 1
            best_row = list(row)
            best_max = max(best_row)
            other_set = set(other_diffs)
            for _ in range(20): # Reduced iterations for speed
                improved = False
                for pi in range(1, n):
                    old_val = best_row[pi]
                    min_allowed = best_row[pi - 1] + 1
                    # Try decreasing
                    for new_val in range(old_val - 1, min_allowed - 1, -1):
                        if new_val < min_allowed: break
                        if pi < n - 1 and new_val >= best_row[pi + 1]: break
                        cand = best_row[:]
                        cand[pi] = new_val
                        valid = True
                        diffs = set()
                        for i in range(n):
                            for j in range(i + 1, n):
                                d = cand[j] - cand[i]
                                if d in diffs:
                                    valid = False
                                    break
                                diffs.add(d)
                        if not valid: continue
                        if diffs & other_set: continue
                        new_max = max(cand)
                        if new_max < best_max:
                            best_row = cand
                            best_max = new_max
                            improved = True
                            break
                        elif new_max == best_max and random.random() < 0.2:
                            best_row = cand
                            improved = True
                            break
                    if improved: break
                if not improved: break
            return best_row

        while time.time() < deadline and no_improve < max_no_improve:
            now = time.time()
            if now - last_reopt_time > reopt_interval and no_improve > 200:
                last_reopt_time = now
                target_row = max(range(n_rows), key=lambda i: row_maxes[i])
                other_diffs = set()
                for j in range(n_rows):
                    if j != target_row:
                        other_diffs |= row_diffs[j]
                improved_row = _reoptimize_row(current_rows[target_row], other_diffs, k, max_scope=best_val)
                if max(improved_row) < row_maxes[target_row]:
                    new_diffs = _row_diffs(improved_row)
                    all_used -= row_diffs[target_row]
                    all_used |= new_diffs
                    current_rows[target_row] = improved_row
                    row_diffs[target_row] = new_diffs
                    row_maxes[target_row] = max(improved_row)
                    current_val = max(row_maxes)
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    continue
            
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
            
            candidates = []
            for d in range(1, min(20, old_val - min_allowed + 1)):
                candidates.append(old_val - d)
            for d in range(1, min(10, max_allowed - old_val + 1)):
                candidates.append(old_val + d)
                
            random.shuffle(candidates)
            
            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]: continue
                if pi < k and new_val >= current_rows[ri][pi + 1]: continue
                
                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                new_diffs = set()
                valid = True
                n = len(cand_row)
                for i in range(n):
                    for j in range(i + 1, n):
                        d = cand_row[j] - cand_row[i]
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                if not valid: continue
                if new_diffs & (all_used - row_diffs[ri]): continue
                
                cand_row_max = max(cand_row)
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                accept = False
                if delta < 0: accept = True
                elif delta == 0 and random.random() < 0.1: accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob: accept = True
                
                if accept:
                    all_used -= row_diffs[ri]
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    moved = True
                    break
            
            if not moved: no_improve += 1
            T = max(T_min, T * cool)
            
        return best_rows, best_val

    # Initial best solution
    best_solution = greedy_construct(n_rows, k)
    best_scope = _global_max(best_solution)
    
    # Run a few SA trials to improve initial bound
    for trial in range(3):
        if time.time() > deadline - 30: break
        seed = trial * 1000 + 42
        init_rows = local_search_init(n_rows, k, seed=seed)
        init_scope = _global_max(init_rows)
        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = init_rows
        
        sa_deadline = time.time() + 10.0
        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, sa_deadline)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows

    # --- Phase 2: Structural Search for Optimal Scope ---
    # We try to reduce the scope by 1, 2, ... using a feasibility check.
    
    def find_min_row(used_diffs, k, max_scope, time_limit):
        """
        Finds a row of length k+1 starting at 0, max <= max_scope,
        with diffs disjoint from used_diffs.
        Returns the row with the minimum possible max value, or None.
        Uses DFS with pruning.
        """
        best_row = None
        best_max = max_scope + 1
        
        def dfs(row, current_max):
            nonlocal best_row, best_max
            if time.time() > time_limit:
                return
            if len(row) == k + 1:
                if current_max < best_max:
                    best_max = current_max
                    best_row = row[:]
                return
            
            last = row[-1]
            # Pruning: even if we pick the smallest possible next values, 
            # can we beat best_max?
            # Minimal increment is 1. So min possible final max is last + (k + 1 - len(row))
            min_possible_final = last + (k + 1 - len(row))
            if min_possible_final >= best_max:
                return
            
            # Existing diffs in row
            existing_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    existing_diffs.add(row[j] - row[i])
            
            # Try next values
            # Heuristic: try small values first to keep max low
            # Upper bound for x: 
            # 1. x < best_max (if we want to improve)
            # 2. x <= max_scope
            # 3. x - a_i not in used_diffs and not in existing_diffs
            
            # We can bound x by best_max - (remaining elements - 1)
            # If we pick x, we need at least (k + 1 - len(row) - 1) more elements, each at least +1.
            # So x + (k + 1 - len(row) - 1) < best_max => x < best_max - (k - len(row))
            max_x_candidate = min(max_scope, best_max - (k - len(row)) + 1) 
            if max_x_candidate < last + 1:
                return
                
            for x in range(last + 1, max_x_candidate + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used_diffs:
                    continue
                if new_diffs & existing_diffs:
                    continue
                
                row.append(x)
                dfs(row, x)
                row.pop()
                
                # If we found a solution, and we are iterating x upwards,
                # any larger x will result in a larger or equal max for this branch.
                # Since we want the MINIMUM max, we can stop searching this branch 
                # if we found a solution with max = x (and x is the current max).
                # Actually, if we found a solution in the subtree of x, 
                # the best_max is updated. 
                # If x >= best_max, we can break.
                if best_max <= x:
                    break

        dfs([0], 0)
        return best_row

    def check_feasibility(n_rows, k, M, time_limit):
        """
        Checks if there exist n_rows disjoint-diff rows with max <= M.
        Returns the list of rows if found, else None.
        Uses backtracking.
        """
        start = time.time()
        
        def backtrack(row_idx, used_diffs, rows_so_far):
            nonlocal best_solution_found
            if time.time() > time_limit:
                return None
            
            if row_idx == n_rows:
                return rows_so_far[:]
            
            # Find a row that fits
            # To allow backtracking, we might want to try multiple rows.
            # For speed, we first try the "best" (min max) row.
            # If that fails, we might need to try others. 
            # Simple strategy: Try min max row. If the whole path fails, 
            # we don't easily backtrack to try the second best row for this index 
            # without a more complex enumeration.
            # However, given the constraints, the "greedy min max" choice is often sufficient 
            # or the failure is due to global incompatibility.
            
            # To be safer, we can generate the top 2-3 candidate rows.
            # But find_min_row only returns 1.
            # Let's modify find_min_row to return a list of top L candidates?
            # For now, let's stick to 1 candidate but rely on the fact that 
            # if we pick the smallest possible row, we leave the most room (largest 
            # available difference space? No, we leave the most *value* space).
            
            row = find_min_row(used_diffs, k, M, time_limit)
            if row is None:
                return None
            
            new_used = used_diffs | _row_diffs(row)
            rows_so_far.append(row)
            result = backtrack(row_idx + 1, new_used, rows_so_far)
            if result is not None:
                return result
            rows_so_far.pop()
            
            # If the greedy choice failed, we could try other rows.
            # To keep it simple and fast, we return None here.
            # This might miss solutions, but it's a strong heuristic.
            # To improve, we could randomize the order of x in find_min_row 
            # or try a few different "best" rows.
            # Let's add a small randomization: if we fail, try a slightly 
            # different search in find_min_row?
            # Actually, let's just return None. The outer loop will try 
            # different M values or rely on the SA solution if this fails.
            
            return None

        rows_so_far = []
        result = backtrack(0, set(), rows_so_far)
        return result

    # Search for better scope
    # We know best_scope is valid. We try best_scope - 1, best_scope - 2, etc.
    # We limit the number of steps to avoid timeout.
    # Each feasibility check has a time limit.
    
    current_scope = best_scope
    steps_to_try = 20 # Try to improve by up to 20
    check_time_limit = 5.0 # 5 seconds per feasibility check
    
    for step in range(steps_to_try):
        if time.time() > deadline - 10:
            break
        
        target_scope = current_scope - 1
        if target_scope < n_rows * k * (k + 1) // 2: # Simple lower bound
            break
            
        # Run feasibility check
        # We use a fresh random seed for any internal randomness if we add it later
        feasible_rows = check_feasibility(n_rows, k, target_scope, time.time() + check_time_limit)
        
        if feasible_rows is not None:
            # Verify the solution
            if _validate(feasible_rows, n_rows, k):
                best_solution = feasible_rows
                best_scope = target_scope
                current_scope = target_scope
                # Continue trying to go lower
            else:
                # Should not happen, but break if invalid
                break
        else:
            # If infeasible, we can't go lower with this method.
            # But wait, the greedy min-max choice might have failed.
            # We could try a randomized version of the check.
            # For now, we assume if the greedy check fails, it's infeasible 
            # or we give up on this specific reduction.
            # To be robust, we might try a few random restarts of the check?
            # Let's try 2 more random checks if the first fails, with a shorter time limit.
            found = False
            for _ in range(2):
                if time.time() > deadline - 10:
                    break
                # Randomize the "min row" selection? 
                # My current find_min_row is deterministic.
                # To randomize, I'd need to shuffle the candidate x's in find_min_row.
                # Let's skip randomization for now to keep code simple and fast.
                pass
            break # Stop searching for lower scope if one step fails

    return best_solution

# EVOLVE-BLOCK-END
