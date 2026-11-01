"""Combined program: Greedy + Backtracking Descent + ILS with Row-Level Backtracking."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin under 150s kill

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
        """All positive differences in a row (as a set)."""
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
    def greedy(jitter=0.0):
        """Greedy construction with optional jitter."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                found = False
                # Look ahead a bit
                for trial_x in range(x + 1, x + 80):
                    new = {trial_x - a for a in row}
                    if not (new & used) and not (new & row_diffs):
                        if jitter > 0 and random.random() < jitter and trial_x < 400:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        used |= new
                        found = True
                        break
                if not found:
                    # Fallback: standard greedy step
                    while True:
                        x += 1
                        new = {x - a for a in row}
                        if not (new & used) and not (new & row_diffs):
                            row.append(x)
                            row_diffs |= new
                            used |= new
                            break
            rows.append(row)
        return rows

    # Initialize with multiple restarts of greedy to find a good starting point
    best_rows = greedy()
    best_scope = scope(best_rows)
    
    if time_left() > 30:
        for _ in range(8):
            if time_left() < 30:
                break
            cand = greedy(jitter=random.uniform(0.05, 0.25))
            if cand is not None:
                s = scope(cand)
                if s < best_scope:
                    best_rows = cand
                    best_scope = s

    # ---------- Row-Level Backtracking Search ----------
    def row_backtrack_search(row_idx, target_max, used_other, time_limit, seed=0):
        """
        Attempts to find a valid row for rows[row_idx] with max value <= target_max.
        Returns the row if found, None otherwise.
        """
        random.seed(seed)
        start = time.time()
        row = [0] * (k + 1)
        
        def solve(pos):
            if time.time() > start + time_limit:
                return False
            
            if pos == k + 1:
                return True
            
            prev_val = row[pos - 1] if pos > 0 else -1
            lower = prev_val + 1
            upper = target_max
            
            if lower > upper:
                return False
            
            # Collect valid candidates
            candidates = []
            # Scan a window first
            scan_limit = min(upper, lower + 50)
            
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - row[i]
                    if d in used_other:
                        valid = False
                        break
                    # Also check against current row diffs (internal to this row)
                    # We need to be careful: used_other is external.
                    # Internal diffs are checked by ensuring d is not in {row[j]-row[i] for i<j<pos}
                    # But we are building sequentially, so we just check against previous values in this row.
                    # Wait, the condition is that ALL differences in the row must be distinct from each other AND from used_other.
                    # So for a new val, d = val - row[i] must not be in used_other.
                    # AND d must not be equal to any other diff in this row.
                    # Since we are adding val, the new diffs are {val - row[i] for i in range(pos)}.
                    # These new diffs must be distinct from each other (which they are, since row is strictly increasing)
                    # AND distinct from existing diffs in this row.
                    # Existing diffs in this row are {row[j] - row[i] for i < j < pos}.
                    
                if valid:
                    # Check against existing diffs in this row
                    existing_diffs = set()
                    for i in range(pos):
                        for j in range(i+1, pos):
                            existing_diffs.add(row[j] - row[i])
                    
                    new_diffs = {val - row[i] for i in range(pos)}
                    if not (new_diffs & existing_diffs):
                        candidates.append(val)
                        
            if not candidates and scan_limit < upper:
                # Scan the rest
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(pos):
                        d = val - row[i]
                        if d in used_other:
                            valid = False
                            break
                    if valid:
                        existing_diffs = set()
                        for i in range(pos):
                            for j in range(i+1, pos):
                                existing_diffs.add(row[j] - row[i])
                        new_diffs = {val - row[i] for i in range(pos)}
                        if not (new_diffs & existing_diffs):
                            candidates.append(val)
                            if len(candidates) >= 10:
                                break
            
            if not candidates:
                return False
            
            # Sort candidates: prefer smaller values to keep row tight
            candidates.sort()
            # Randomize a bit: shuffle the first few
            if len(candidates) > 3:
                random.shuffle(candidates[:3])
            
            for val in candidates:
                row[pos] = val
                if solve(pos + 1):
                    return True
                row[pos] = 0
            
            return False

        if solve(1):
            return row[:]
        return None

    # ---------- Global Backtracking Search with Target Scope ----------
    def global_backtrack_search(target_scope, time_limit, seed=0):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        """
        random.seed(seed)
        start = time.time()
        
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        used_diffs = set()
        
        def solve(row_idx, pos):
            if time.time() > start + time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == k + 1:
                return solve(row_idx + 1, 1)
            
            prev_val = rows[row_idx][pos - 1] if pos > 0 else -1
            lower = prev_val + 1
            upper = target_scope
            
            if lower > upper:
                return False
            
            candidates = []
            scan_limit = min(upper, lower + 50)
            
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 10:
                        break
            
            if not candidates:
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(pos):
                        d = val - rows[row_idx][i]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(val)
                        if len(candidates) >= 10:
                            break
            
            if not candidates:
                return False
            
            candidates.sort()
            if len(candidates) > 3:
                random.shuffle(candidates[:3])
            
            for val in candidates:
                new_diffs = []
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    new_diffs.append(d)
                    used_diffs.add(d)
                
                rows[row_idx][pos] = val
                
                if solve(row_idx, pos + 1):
                    return True
                
                rows[row_idx][pos] = 0
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False

        if solve(0, 1):
            return [r[:] for r in rows]
        return None

    # Phase 1: Global Backtracking Descent
    current_target = best_scope - 1
    seed_counter = 0
    consecutive_failures = 0
    
    while time_left() > 30 and current_target > 100:
        if consecutive_failures > 3:
            break
            
        time_budget = 3.0
        if time_left() > 60 and current_target == best_scope - 1:
            time_budget = 10.0

        found = global_backtrack_search(current_target, time_budget, seed=seed_counter)
        seed_counter += 1
        
        if found:
            best_rows = found
            best_scope = current_target
            consecutive_failures = 0
            current_target -= 1
        else:
            consecutive_failures += 1
            if consecutive_failures == 1:
                 found = global_backtrack_search(current_target, time_budget, seed=seed_counter)
                 seed_counter += 1
                 if found:
                     best_rows = found
                     best_scope = current_target
                     consecutive_failures = 0
                     current_target -= 1
            else:
                 current_target -= 1

    # ---------- Per-row local optimization using Row-Level Backtracking ----------
    def used_set_except(rows, idx):
        used = set()
        for j in range(n_rows):
            if j != idx:
                used |= row_diffs_set(rows[j])
        return used

    def try_reduce_row_backtrack(rows, row_idx, used_other, target_max, time_limit, seed=0):
        """Try to find a row at row_idx with max entry <= target_max."""
        return row_backtrack_search(row_idx, target_max, used_other, time_limit, seed)

    def local_polish_backtrack(rows):
        """Reduce scope by locally re-optimizing each row using backtracking."""
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 5
        pass_count = 0
        while improved and time_left() > 10 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 10:
                    break
                current_max = max(rows[i])
                # Try to reduce max by 1
                target_max = current_max - 1
                if target_max < k + 1: # Must have at least k+1 distinct integers starting at 0
                    continue
                
                used_other = used_set_except(rows, i)
                # Allocate a small time budget for this row
                t_budget = 1.0
                if time_left() > 50:
                    t_budget = 2.0
                
                new_row = try_reduce_row_backtrack(rows, i, used_other, target_max, t_budget, seed=random.randint(0, 10000))
                if new_row is not None:
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    # Phase 2: Initial Polish with Backtracking
    if time_left() > 15:
        polished = local_polish_backtrack(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Phase 3: Iterated Local Search with Backtracking Polish
    def kick(rows):
        """Perturb the solution to escape local minima."""
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            used_other = used_set_except(rows, idx)
            # Rebuild with greedy + jitter
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
                # Fallback to greedy
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

    while time_left() > 20:
        # 1. Local Search (Backtrack Polish)
        polished = local_polish_backtrack(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        # 2. Kick
        if time_left() < 15:
            break
        kicked_rows = kick(best_rows)
        
        # 3. Local Search on kicked rows
        polished_kicked = local_polish_backtrack(kicked_rows)
        
        # 4. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Final polish
    if time_left() > 5:
        polished = local_polish_backtrack(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity guarantee.
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
