import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid approach: greedy initialization, then randomized local search
    with simulated annealing to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # Lower bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed  # scope must be at least this

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

    # ---- Randomized Local Search with Simulated Annealing ----
    # This is the main workhorse for finding good solutions

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

    def simulated_annealing(rows, n_rows, k, start_time, deadline):
        """SA-based local search to minimize the maximum entry."""
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        
        T = 300.0
        T_min = 0.1
        cool = 0.99995
        
        no_improve = 0
        max_no_improve = 50000
        restarts = 0
        max_restarts = 30
        
        # Adaptive step size based on how stuck we are
        step_scale = 1.0
        
        # Find which row(s) have the max value
        def _max_rows():
            return [i for i in range(n_rows) if row_maxes[i] == current_val]
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Bias towards rows with the maximum value
            max_r = _max_rows()
            if max_r and random.random() < 0.7:
                ri = random.choice(max_r)
            else:
                ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            # Generate candidates - prioritize decreasing moves
            candidates = []
            
            # Adaptive step sizes
            dec_range = int(min(100 * step_scale, old_val - current_rows[ri][pi - 1] - 1))
            inc_range = int(min(40 * step_scale, (current_rows[ri][pi + 1] - 1 if pi < k else current_val + 60) - old_val))
            
            # Decreasing candidates (primary optimization direction)
            for d in range(1, dec_range + 1):
                candidates.append(old_val - d)
            
            # Increasing candidates (exploration)
            for d in range(1, inc_range + 1):
                candidates.append(old_val + d)
            
            # Random jumps
            if random.random() < 0.4:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 50
                if low <= high:
                    candidates.append(random.randint(low, high))
            
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                step_scale = max(0.5, step_scale * 0.95)
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
                
                # Check cross-row conflicts
                for d in new_diffs:
                    if d in all_used and d not in row_diffs[ri]:
                        valid = False
                        break
                
                if not valid:
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
                    if random.random() < 0.3:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.05))
                        if random.random() < prob:
                            accept = True
                
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
                        step_scale = min(2.0, step_scale * 1.1)
                    else:
                        no_improve += 1
                    
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
                step_scale = max(0.3, step_scale * 0.97)
            
            T = max(T_min, T * cool)
            
            # Restart if stuck
            if no_improve > max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                current_rows = [r[:] for r in best_rows]
                current_val = best_val
                all_used = _all_used(current_rows)
                row_diffs = [_row_diffs(r) for r in current_rows]
                row_maxes = [max(r) for r in current_rows]
                T = max(200.0, T * 2)
                no_improve = 0
                step_scale = 1.0
        
        return best_rows, best_val

    # Run multiple SA trials with different seeds
    for trial in range(10):
        if time.time() > deadline - 5:
            break
        
        seed = trial * 7919 + 12345
        init_rows = local_search_init(n_rows, k, seed=seed)
        init_scope = _global_max(init_rows)
        
        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]
        
        # Give each trial a fair time budget
        trial_deadline = min(deadline - 5, start_time + 140.0)
        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, start_time, trial_deadline)
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows
    
    # Final deterministic polish: try all single-element decreases
    if time.time() < deadline - 2:
        polish_deadline = min(deadline, start_time + 143.0)
        polish_rows, polish_scope = simulated_annealing(
            [r[:] for r in best_solution], n_rows, k, start_time, polish_deadline
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows
        
        # Deterministic local improvement: greedily try to decrease each element
        if time.time() < deadline - 1:
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                for ri in range(n_rows):
                    for pi in range(1, k + 1):
                        if time.time() > deadline - 1:
                            break
                        old_val = best_solution[ri][pi]
                        # Try to decrease by 1, 2, 3, ...
                        for dec in range(1, min(20, old_val - best_solution[ri][pi-1])):
                            new_val = old_val - dec
                            if new_val <= best_solution[ri][pi - 1]:
                                break
                            if pi < k and new_val >= best_solution[ri][pi + 1]:
                                break
                            
                            cand_row = best_solution[ri][:]
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
                            
                            # Check cross-row conflicts
                            conflict = False
                            for other_ri in range(n_rows):
                                if other_ri == ri:
                                    continue
                                for d in new_diffs:
                                    if d in _row_diffs(best_solution[other_ri]):
                                        conflict = True
                                        break
                                if conflict:
                                    break
                            
                            if conflict:
                                continue
                            
                            # Accept if it improves the scope
                            cand_row_max = max(cand_row)
                            old_row_max = max(best_solution[ri])
                            if cand_row_max < old_row_max or cand_row_max <= best_scope:
                                # Check if this reduces overall max
                                new_maxes = [max(r) for r in best_solution]
                                new_maxes[ri] = cand_row_max
                                if max(new_maxes) <= best_scope:
                                    best_solution[ri] = cand_row
                                    best_scope = max(new_maxes)
                                    improved = True
                                    break
            
            # Final validation
            if not _validate(best_solution, n_rows, k):
                # Fall back to greedy if something went wrong
                best_solution = greedy_construct(n_rows, k)
                best_scope = max(max(row) for row in best_solution)

    return best_solution
# EVOLVE-BLOCK-END
