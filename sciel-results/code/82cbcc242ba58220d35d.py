import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid approach: greedy initialization, then simulated annealing with
    row-level operations (row re-solve, row swap, position moves) to minimize
    the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # ---- Helper functions ----
    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _all_used(rows):
        all_used = set()
        for row in rows:
            all_used |= _row_diffs(row)
        return all_used

    def _global_max(rows):
        return max(max(row) for row in rows)

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

    def greedy_construct(n_rows, k, scope_limit=None):
        """Greedy construction with optional scope limit."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if scope_limit is not None and x > scope_limit:
                    return None
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # ---- Greedy fallback (always valid) ----
    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    # ---- Row-level operations for SA ----
    def _try_row_resolved(row, used_excluding_row, scope_limit):
        """
        Try to re-solve a row given the differences used by other rows.
        Returns a new row or None if impossible within scope_limit.
        """
        # We need to find k+1 elements starting at 0 such that all differences
        # are not in used_excluding_row
        # Use greedy with backtracking
        def _backtrack_solve(row_so_far, used_local, x_start):
            if len(row_so_far) == k + 1:
                return row_so_far[:]
            
            x = x_start
            while True:
                x += 1
                if scope_limit is not None and x > scope_limit:
                    return None
                
                new_diffs = {x - a for a in row_so_far}
                # Check against local used and global used
                if new_diffs & used_local:
                    continue
                if new_diffs & used_excluding_row:
                    continue
                
                # Check internal consistency
                row_so_far.append(x)
                used_local |= new_diffs
                
                result = _backtrack_solve(row_so_far, used_local, x)
                if result is not None:
                    return result
                
                row_so_far.pop()
                used_local -= new_diffs
                
                if x > scope_limit if scope_limit else False:
                    break
            
            return None
        
        return _backtrack_solve([0], set(), 0)

    def _row_swap_feasible(row_i, row_j, all_used, row_diffs_i, row_diffs_j):
        """Check if swapping two rows is feasible (it always is, since we just exchange)."""
        # Swapping rows doesn't change the set of used differences
        return True

    def simulated_annealing(rows, n_rows, k, start_time, deadline):
        """SA-based local search with row-level operations."""
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        
        T = 500.0
        T_min = 0.5
        cool = 0.99998
        
        no_improve = 0
        max_no_improve = 25000
        restarts = 0
        max_restarts = 15
        
        def _max_rows():
            return [i for i in range(n_rows) if row_maxes[i] == current_val]
        
        def _try_position_move(ri, pi, T):
            """Try a single-position move."""
            old_val = current_rows[ri][pi]
            
            candidates = []
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(60, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
            
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
            max_increase = min(20, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
            
            if random.random() < 0.3:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
                if low <= high:
                    candidates.append(random.randint(low, high))
            
            if not candidates:
                return None, 0, 0
            
            candidates.sort(key=lambda v: (v > old_val, v))
            
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                
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
                
                if not valid:
                    continue
                
                for d in new_diffs:
                    if d in all_used and d not in row_diffs[ri]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                cand_row_max = max(cand_row)
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.2:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    all_used -= row_diffs[ri]
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)
                    return True, delta, new_global_max
            
            return None, 0, 0

        def _try_row_swap(ri, rj):
            """Try swapping two rows."""
            if ri == rj:
                return None, 0
            
            # Check if swap improves things (it shouldn't change the global max
            # unless the rows have different maxes)
            max_i = row_maxes[ri]
            max_j = row_maxes[rj]
            
            # Swapping doesn't change the set of used differences
            # But it might help if we then do position moves on the swapped rows
            # For now, just swap and see
            current_rows[ri], current_rows[rj] = current_rows[rj], current_rows[ri]
            row_maxes[ri], row_maxes[rj] = row_maxes[rj], row_maxes[ri]
            row_diffs[ri], row_diffs[rj] = row_diffs[rj], row_diffs[ri]
            
            new_global_max = max(row_maxes)
            delta = new_global_max - current_val
            
            # Accept swap with some probability to explore
            if delta <= 0 or random.random() < 0.3:
                current_val = new_global_max
                return True, delta, new_global_max
            else:
                # Revert
                current_rows[ri], current_rows[rj] = current_rows[rj], current_rows[ri]
                row_maxes[ri], row_maxes[rj] = row_maxes[rj], row_maxes[ri]
                row_diffs[ri], row_diffs[rj] = row_diffs[rj], row_diffs[ri]
                return None, 0, 0

        def _try_row_resolved(ri, T):
            """Try to re-solve a row to minimize its max."""
            used_excl = all_used - row_diffs[ri]
            old_max = row_maxes[ri]
            
            # Try to find a better row
            new_row = _try_row_resolved(current_rows[ri], used_excl, old_max - 1)
            
            if new_row is not None:
                new_diffs = _row_diffs(new_row)
                new_max = max(new_row)
                
                # Validate no conflicts
                valid = True
                for d in new_diffs:
                    if d in used_excl:
                        valid = False
                        break
                
                if valid:
                    delta = max(row_maxes[:ri] + [new_max] + row_maxes[ri+1:]) - current_val
                    accept = False
                    if delta < 0:
                        accept = True
                    elif delta == 0:
                        if random.random() < 0.2:
                            accept = True
                    else:
                        if T > T_min:
                            prob = math.exp(-delta / max(T, 0.1))
                            if random.random() < prob:
                                accept = True
                    
                    if accept:
                        all_used -= row_diffs[ri]
                        all_used |= new_diffs
                        current_rows[ri] = new_row
                        row_diffs[ri] = new_diffs
                        row_maxes[ri] = new_max
                        current_val = max(row_maxes)
                        return True, delta, current_val
            
            return None, 0, 0

        while time.time() < deadline and no_improve < max_no_improve:
            # Choose operation type
            op = random.random()
            
            if op < 0.5:
                # Position move
                max_r = _max_rows()
                if max_r and random.random() < 0.6:
                    ri = random.choice(max_r)
                else:
                    ri = random.randrange(n_rows)
                pi = random.randrange(1, k + 1)
                
                result, delta, new_val = _try_position_move(ri, pi, T)
                
                if result is True:
                    if new_val < best_val:
                        best_val = new_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            elif op < 0.7:
                # Row swap
                ri = random.randrange(n_rows)
                rj = random.randrange(n_rows)
                if ri != rj:
                    result, delta, new_val = _try_row_swap(ri, rj)
                    if result is True:
                        if new_val < best_val:
                            best_val = new_val
                            best_rows = [r[:] for r in current_rows]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            else:
                # Row re-solve
                ri = random.randrange(n_rows)
                result, delta, new_val = _try_row_resolved(ri, T)
                if result is True:
                    if new_val < best_val:
                        best_val = new_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            T = max(T_min, T * cool)
            
            if no_improve > max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                current_rows = [r[:] for r in best_rows]
                current_val = best_val
                all_used = _all_used(current_rows)
                row_diffs = [_row_diffs(r) for r in current_rows]
                row_maxes = [max(r) for r in current_rows]
                T = 300.0
                no_improve = 0
        
        return best_rows, best_val

    # Run multiple SA trials with different seeds
    for trial in range(8):
        if time.time() > deadline - 10:
            break
        
        seed = trial * 7919 + 12345
        random.seed(seed)
        init_rows = greedy_construct(n_rows, k)
        init_scope = _global_max(init_rows)
        
        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]
        
        trial_deadline = min(deadline - 10, start_time + 140.0)
        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, start_time, trial_deadline)
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows

    # Final polish with remaining time
    if time.time() < deadline - 5:
        polish_deadline = min(deadline, start_time + 143.0)
        polish_rows, polish_scope = simulated_annealing(
            [r[:] for r in best_solution], n_rows, k, start_time, polish_deadline
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows

    return best_solution
# EVOLVE-BLOCK-END
