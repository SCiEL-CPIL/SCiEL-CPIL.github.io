import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid approach: greedy initialization, then randomized local search
    with simulated annealing and row-level re-optimization to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # ---- Greedy fallback (always valid) ----
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

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

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

    def local_search_init(n_rows, k, seed=None):
        """Greedy initialization with randomization."""
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

    def _reoptimize_row(row, other_diffs, k, max_scope=None):
        """
        Globally re-optimize a single row against fixed differences from other rows.
        Uses iterative local search on the row elements to minimize the max entry
        while avoiding conflicts with other_diffs and internal conflicts.
        Returns the improved row.
        """
        n = k + 1
        best_row = list(row)
        best_max = max(best_row)
        
        if max_scope is not None and best_max > max_scope:
            return best_row
        
        other_set = set(other_diffs)
        
        for iteration in range(120):
            improved = False
            for pi in range(1, n):
                old_val = best_row[pi]
                min_allowed = best_row[pi - 1] + 1
                max_allowed = (best_row[pi + 1] - 1) if pi < n - 1 else (max_scope - 1 if max_scope else old_val)
                
                # Try decreasing values first
                for new_val in range(old_val - 1, min_allowed - 1, -1):
                    if new_val < min_allowed:
                        break
                    if pi < n - 1 and new_val >= best_row[pi + 1]:
                        break
                    cand = best_row[:]
                    cand[pi] = new_val
                    # Check internal validity
                    valid = True
                    diffs = set()
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = cand[j] - cand[i]
                            if d in diffs:
                                valid = False
                                break
                            diffs.add(d)
                    if not valid:
                        continue
                    # Check against other rows' differences
                    if diffs & other_set:
                        continue
                    # Accept if it reduces the max
                    new_max = max(cand)
                    if new_max < best_max:
                        best_row = cand
                        best_max = new_max
                        improved = True
                        break
                    elif new_max == best_max:
                        # Accept neutral moves to escape traps
                        if random.random() < 0.4:
                            best_row = cand
                            improved = True
                            break
            if not improved:
                break
        return best_row

    def simulated_annealing(rows, n_rows, k, start_time, deadline):
        """SA-based local search with row-level re-optimization to minimize the maximum entry."""
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        row_diffs = [_row_diffs(r) for r in current_rows]
        
        # Pre-compute other_diffs for each row (differences from all other rows)
        other_diffs = [set() for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_rows):
                if i != j:
                    other_diffs[i] |= row_diffs[j]
        
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        
        T = 500.0
        T_min = 0.5
        cool = 0.99998
        
        no_improve = 0
        max_no_improve = 30000
        restarts = 0
        max_restarts = 20
        last_reopt_time = time.time()
        reopt_interval = 1.0  # seconds between row re-optimization attempts
        
        # Find which row(s) have the max value
        def _max_rows():
            return [i for i in range(n_rows) if row_maxes[i] == current_val]
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Periodically attempt row-level re-optimization
            now = time.time()
            if now - last_reopt_time > reopt_interval and no_improve > 200:
                last_reopt_time = now
                # Try re-optimizing the row with the largest max
                target_row = max(range(n_rows), key=lambda i: row_maxes[i])
                improved_row = _reoptimize_row(current_rows[target_row], other_diffs[target_row], k, max_scope=best_val)
                if max(improved_row) < row_maxes[target_row]:
                    # Apply the improvement
                    new_diffs = _row_diffs(improved_row)
                    old_diffs = row_diffs[target_row]
                    current_rows[target_row] = improved_row
                    row_diffs[target_row] = new_diffs
                    row_maxes[target_row] = max(improved_row)
                    current_val = max(row_maxes)
                    
                    # Update other_diffs for all other rows
                    for j in range(n_rows):
                        if j != target_row:
                            other_diffs[j] -= old_diffs
                            other_diffs[j] |= new_diffs
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    continue
            
            # Bias towards rows with the maximum value
            max_r = _max_rows()
            if max_r and random.random() < 0.6:
                ri = random.choice(max_r)
            else:
                ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            # Generate candidates - prioritize decreasing moves
            candidates = []
            
            # Decreasing candidates (primary optimization direction)
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(80, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
            
            # Increasing candidates (exploration)
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 50
            max_increase = min(20, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
            
            # Random jumps
            if random.random() < 0.3:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
                if low <= high:
                    candidates.append(random.randint(low, high))
            
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            # Sort candidates: decreasing moves first (lower values preferred)
            candidates.sort(key=lambda v: (v > old_val, v))
            
            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                
                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                
                # Validate
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
                
                # Check cross-row conflicts using pre-computed other_diffs
                if new_diffs & other_diffs[ri]:
                    continue
                
                # Compute new global max
                cand_row_max = max(cand_row)
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                
                # Acceptance criterion
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
                    old_diffs = row_diffs[ri]
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)
                    
                    # Update other_diffs for all other rows
                    for j in range(n_rows):
                        if j != ri:
                            other_diffs[j] -= old_diffs
                            other_diffs[j] |= new_diffs
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
            
            T = max(T_min, T * cool)
            
            # Restart if stuck
            if no_improve > max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                current_rows = [r[:] for r in best_rows]
                current_val = best_val
                row_diffs = [_row_diffs(r) for r in current_rows]
                row_maxes = [max(r) for r in current_rows]
                # Rebuild other_diffs
                other_diffs = [set() for _ in range(n_rows)]
                for i in range(n_rows):
                    for j in range(n_rows):
                        if i != j:
                            other_diffs[i] |= row_diffs[j]
                T = 300.0
                no_improve = 0
                last_reopt_time = time.time()
        
        return best_rows, best_val

    # ---- Final polish: high-precision local search with tighter parameters ----
    def final_polish(rows, n_rows, k, deadline):
        """
        High-precision polish pass: use tighter tolerances, more iterations,
        and exact local solves to squeeze out the last improvements.
        """
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        row_diffs = [_row_diffs(r) for r in current_rows]
        
        # Pre-compute other_diffs
        other_diffs = [set() for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_rows):
                if i != j:
                    other_diffs[i] |= row_diffs[j]
        
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        
        T = 100.0
        T_min = 0.1
        cool = 0.99995
        
        no_improve = 0
        max_no_improve = 50000
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Row-level re-optimization with tighter scope
            target_row = max(range(n_rows), key=lambda i: row_maxes[i])
            improved_row = _reoptimize_row(current_rows[target_row], other_diffs[target_row], k, max_scope=best_val)
            if max(improved_row) < row_maxes[target_row]:
                new_diffs = _row_diffs(improved_row)
                old_diffs = row_diffs[target_row]
                current_rows[target_row] = improved_row
                row_diffs[target_row] = new_diffs
                row_maxes[target_row] = max(improved_row)
                current_val = max(row_maxes)
                
                # Update other_diffs
                for j in range(n_rows):
                    if j != target_row:
                        other_diffs[j] -= old_diffs
                        other_diffs[j] |= new_diffs
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                continue
            
            # Fine-grained SA moves - bias toward max row
            max_r = [i for i in range(n_rows) if row_maxes[i] == current_val]
            if max_r and random.random() < 0.5:
                ri = random.choice(max_r)
            else:
                ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            min_allowed = current_rows[ri][pi - 1] + 1
            
            # Try small decreasing moves
            moved = False
            for d in range(1, min(15, old_val - min_allowed + 1)):
                new_val = old_val - d
                if new_val <= min_allowed:
                    break
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    break
                
                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                
                new_diffs = set()
                valid = True
                n = len(cand_row)
                for i in range(n):
                    for j in range(i + 1, n):
                        dd = cand_row[j] - cand_row[i]
                        if dd in new_diffs:
                            valid = False
                            break
                        new_diffs.add(dd)
                
                if not valid:
                    continue
                
                if new_diffs & other_diffs[ri]:
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
                    if random.random() < 0.1:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.05))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    old_diffs = row_diffs[ri]
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)
                    
                    # Update other_diffs
                    for j in range(n_rows):
                        if j != ri:
                            other_diffs[j] -= old_diffs
                            other_diffs[j] |= new_diffs
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
            
            T = max(T_min, T * cool)
        
        return best_rows, best_val

    # Run multiple SA trials with different seeds
    n_trials = 10
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        seed = trial * 7919 + 12345
        init_rows = local_search_init(n_rows, k, seed=seed)
        init_scope = _global_max(init_rows)
        
        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]
        
        # Give each trial a fair time budget
        remaining = deadline - time.time()
        if trial == 0:
            trial_deadline = time.time() + min(remaining * 0.30, 40.0)
        elif trial == 1:
            trial_deadline = time.time() + min(remaining * 0.22, 30.0)
        elif trial == 2:
            trial_deadline = time.time() + min(remaining * 0.18, 22.0)
        else:
            trial_deadline = time.time() + min(remaining * 0.12, 18.0)
        
        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, start_time, trial_deadline)
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows
    
    # Final polish with remaining time
    if time.time() < deadline - 2:
        polish_deadline = min(deadline, start_time + 143.0)
        polish_rows, polish_scope = final_polish(
            [r[:] for r in best_solution], n_rows, k, polish_deadline
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows
    
    # Final row-level re-optimization pass
    if time.time() < deadline - 1:
        for _ in range(5):
            if time.time() > deadline - 1:
                break
            improved_any = False
            for ri in range(n_rows):
                if time.time() > deadline - 1:
                    break
                other_diffs = set()
                for j in range(n_rows):
                    if j != ri:
                        other_diffs |= _row_diffs(best_solution[j])
                improved_row = _reoptimize_row(best_solution[ri], other_diffs, k, max_scope=best_scope)
                if max(improved_row) < max(best_solution[ri]):
                    best_solution[ri] = improved_row
                    best_scope = max(max(row) for row in best_solution)
                    improved_any = True
            if not improved_any:
                break

    return best_solution
# EVOLVE-BLOCK-END
