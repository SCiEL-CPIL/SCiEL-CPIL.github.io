"""Combined program: Greedy + ILS with row-swap moves and targeted backtracking."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy(seed=0):
        """
        Randomized greedy construction.
        If seed > 0, introduces jitter to explore different solutions.
        """
        random.seed(seed) if seed > 0 else None
        used = set()
        rows = []
        for r_idx in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                found = False
                window = 100 if (seed > 0 and r_idx > 0) else 50
                for trial_x in range(x + 1, x + window):
                    new = {trial_x - a for a in row}
                    if not (new & used) and not (new & row_diffs):
                        if seed > 0 and random.random() < 0.1 and trial_x < 200:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        used |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(x + 1, x + 500):
                        new = {trial_x - a for a in row}
                        if not (new & used) and not (new & row_diffs):
                            row.append(trial_x)
                            row_diffs |= new
                            used |= new
                            found = True
                            break
                if not found:
                    break
                x = row[-1]
            if len(row) == k + 1:
                rows.append(row)
            else:
                pass
        
        if len(rows) != n_rows:
            used = set()
            rows = []
            for _ in range(n_rows):
                row, x = [0], 0
                row_diffs = set()
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used) and not (new & row_diffs):
                        row.append(x)
                        row_diffs |= new
                        used |= new
                rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Per-row local optimization ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=150):
        """Try to find a row at row_idx with smaller max entry."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            jitter = 0.10 + 0.05 * min(attempt / max_attempts, 1.0)
            while len(row) < k + 1:
                found = False
                window = 80 + 20 * min(attempt / max_attempts, 1.0)
                for trial_x in range(x + 1, int(x + window)):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        if jitter > 0 and random.random() < jitter and trial_x < 300:
                            continue
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(x + 1, x + 200):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            row.append(trial_x)
                            current_row_diffs |= new
                            found = True
                            break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) < best_max:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
        return best_row, best_max

    def try_swap_rows(rows, i, j, max_attempts=50):
        """Try to rebuild row i given row j's diffs are also in use, then swap."""
        # This is a simplified swap: rebuild both rows i and j simultaneously
        # by trying to find two rows that fit better
        pass  # We'll implement a proper swap below

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 6
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=80)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def kick(rows):
        """Perturb the solution to escape local minima."""
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            row, x = [0], 0
            current_row_diffs = set()
            success = False
            for attempt in range(25):
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                while len(row) < k + 1:
                    found = False
                    for trial_x in range(x + 1, x + 120):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            if random.random() < 0.45 and trial_x < 350:
                                continue
                            row.append(trial_x)
                            current_row_diffs |= new
                            found = True
                            break
                    if not found:
                        valid_row = False
                        break
                    x = row[-1]
                
                if valid_row:
                    rows[idx] = row
                    success = True
                    break
            
            if not success:
                greedy_row, x = [0], 0
                row_diffs = set()
                while len(greedy_row) < k + 1:
                    x += 1
                    new = {x - a for a in greedy_row}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(x)
                        row_diffs |= new
                rows[idx] = greedy_row

        return rows

    # ---------- Targeted Backtracking Search ----------
    def backtrack_search(target_scope, time_limit):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        """
        start = time.time()
        
        def solve_bt():
            rows_local = []
            used_local = set()
            nodes_visited = 0
            max_nodes = 500000
            
            def rec(row_idx, curr_row, curr_diffs):
                nonlocal nodes_visited
                if time.time() - start > time_limit or nodes_visited > max_nodes:
                    return False
                
                if len(curr_row) == k + 1:
                    rows_local.append(curr_row)
                    used_local.update(curr_diffs)
                    
                    if row_idx == n_rows - 1:
                        return True
                    
                    return rec(row_idx + 1, [0], set())
                
                last_val = curr_row[-1]
                candidates = list(range(last_val + 1, target_scope + 1))
                
                def conflict_score(val):
                    new_diffs = {val - a for a in curr_row}
                    conflicts = len(new_diffs & used_local) + len(new_diffs & curr_diffs)
                    return (conflicts, val)
                
                if random.random() < 0.3:
                    random.shuffle(candidates)
                else:
                    candidates.sort(key=conflict_score)
                
                max_candidates = min(len(candidates), 150)
                candidates = candidates[:max_candidates]
                
                for val in candidates:
                    if time.time() - start > time_limit or nodes_visited > max_nodes:
                        return False
                    
                    nodes_visited += 1
                    
                    new_diffs = set()
                    valid = True
                    for a in curr_row:
                        d = val - a
                        if d in used_local or d in curr_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        if rec(row_idx, curr_row + [val], curr_diffs | new_diffs):
                            return True
                
                return False
            
            if rec(0, [0], set()):
                return rows_local
            return None

        return solve_bt()

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Phase 1: Randomized Greedy Restarts
    num_restarts = 15
    for i in range(num_restarts):
        if time_left() < 20:
            break
        
        new_rows = greedy(seed=i+1)
        if new_rows:
            new_scope = scope(new_rows)
            if new_scope < best_scope:
                best_rows = new_rows
                best_scope = new_scope
                if time_left() > 15:
                    polished = local_polish(best_rows)
                    if scope(polished) < best_scope:
                        best_rows = polished
                        best_scope = scope(polished)

    # Phase 2: ILS on the best found so far
    while time_left() > 30:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        if time_left() < 25:
            break
            
        kicked_rows = kick(best_rows)
        
        polished_kicked = local_polish(kicked_rows)
        
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Phase 3: Targeted Backtracking
    if time_left() > 15:
        target = best_scope - 1
        bt_rows = backtrack_search(target, time_limit=12)
        if bt_rows:
            best_rows = bt_rows
            best_scope = target
            if time_left() > 5:
                polished = local_polish(best_rows)
                if scope(polished) < best_scope:
                    best_rows = polished
                    best_scope = scope(polished)
        
        if time_left() > 12:
            target = best_scope - 1
            bt_rows = backtrack_search(target, time_limit=10)
            if bt_rows:
                best_rows = bt_rows
                best_scope = target
                if time_left() > 5:
                    polished = local_polish(best_rows)
                    if scope(polished) < best_scope:
                        best_rows = polished
                        best_scope = scope(polished)
        
        if time_left() > 10:
            target = best_scope - 1
            bt_rows = backtrack_search(target, time_limit=8)
            if bt_rows:
                best_rows = bt_rows
                best_scope = target
                if time_left() > 5:
                    polished = local_polish(best_rows)
                    if scope(polished) < best_scope:
                        best_rows = polished
                        best_scope = scope(polished)

    # Final high-precision polish
    if time_left() > 8:
        best_rows = [list(r) for r in best_rows]
        for _ in range(3):
            if time_left() < 3:
                break
            improved = False
            order = sorted(range(n_rows), key=lambda i: -max(best_rows[i]))
            for i in order:
                if time_left() < 2:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(best_rows[j])
                new_row, new_max = try_reduce_row(best_rows, i, used_other, max_attempts=200)
                if new_max < max(best_rows[i]):
                    best_rows[i] = new_row
                    improved = True
            if not improved:
                break
        
        if scope(best_rows) < best_scope:
            best_scope = scope(best_rows)

    # Final validity guarantee.
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
