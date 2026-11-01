import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 138.0

    def time_left():
        return deadline - time.time()

    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def row_diffs_list(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
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

    def total_conflicts(rows):
        """Count total duplicate differences across all rows."""
        seen = {}
        total = 0
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    total += 1
                else:
                    seen[d] = True
        return total

    # ---------- Greedy baseline ----------
    def greedy():
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

    # ---------- Row generation with randomization ----------
    def generate_row(used_other, max_val, randomness=0.0):
        """Generate a row of k+1 increasing integers starting at 0.
        randomness: probability of skipping a valid candidate to explore."""
        row, x = [0], 0
        row_diffs = set()
        while len(row) < k + 1:
            found = False
            # Search in a window
            window_end = min(x + 120, max_val)
            candidates = []
            for trial_x in range(x + 1, window_end + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    candidates.append(trial_x)
            if not candidates:
                # Try harder: extend window
                for trial_x in range(x + 1, max_val + 1):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & row_diffs):
                        candidates.append(trial_x)
                        break
            if not candidates:
                return None
            if randomness > 0 and len(candidates) > 1 and random.random() < randomness:
                # Pick a random candidate from the first several
                pick = random.choice(candidates[:min(5, len(candidates))])
            else:
                pick = candidates[0]
            row.append(pick)
            new = {pick - a for a in row[:-1]}
            row_diffs |= new
            x = pick
        return row

    # ---------- Simulated Annealing ----------
    def anneal(rows, time_budget):
        """Simulated annealing: try replacing random rows with new generations."""
        rows = [list(r) for r in rows]
        current_scope = scope(rows)
        best_rows_sa = [list(r) for r in rows]
        best_scope_sa = current_scope
        
        # Initial temperature based on scope
        T = current_scope * 0.3
        T_min = 0.1
        cooling_rate = 0.995
        
        used_other_cache = {}
        for i in range(n_rows):
            used_other = set()
            for j in range(n_rows):
                if j != i:
                    used_other |= row_diffs_set(rows[j])
            used_other_cache[i] = used_other
        
        end_time = time.time() + time_budget
        while time.time() < end_time and T > T_min:
            T *= cooling_rate
            if T < T_min:
                break
            
            # Pick a random row to replace
            idx = random.randint(0, n_rows - 1)
            used_other = used_other_cache[idx]
            
            # Generate a new row
            randomness = min(0.5, T / (current_scope * 0.3))
            new_row = generate_row(used_other, current_scope + 20, randomness)
            if new_row is None:
                continue
            
            # Evaluate
            test_rows = [list(r) for r in rows]
            test_rows[idx] = new_row
            test_scope = scope(test_rows)
            
            # Check validity
            if not is_valid(test_rows):
                # Compute conflict penalty
                conflict_count = total_conflicts(test_rows)
                delta_energy = conflict_count * 5 + (test_scope - current_scope)
            else:
                delta_energy = test_scope - current_scope
            
            # Acceptance decision
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / max(T, 0.01)):
                # Update used_other_cache
                old_diffs = row_diffs_set(rows[idx])
                rows[idx] = new_row
                new_diffs = row_diffs_set(new_row)
                for j in range(n_rows):
                    if j != idx:
                        used_other_cache[j] -= old_diffs
                        used_other_cache[j] |= new_diffs
                current_scope = scope(rows)
                
                if current_scope < best_scope_sa:
                    best_scope_sa = current_scope
                    best_rows_sa = [list(r) for r in rows]
            
            # Rebuild cache if needed (simplified: just recompute for the row we changed)
            # Actually, let's just recompute all caches periodically
            if random.random() < 0.05:
                for j in range(n_rows):
                    used_other_cache[j] = set()
                    for m in range(n_rows):
                        if m != j:
                            used_other_cache[j] |= row_diffs_set(rows[m])
        
        return best_rows_sa, best_scope_sa

    # ---------- Local polish ----------
    def local_polish(rows):
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 8
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
                
                # Try to find a better row
                current_max = max(rows[i])
                best_row = list(rows[i])
                best_max = current_max
                
                for attempt in range(100):
                    if time_left() < 3:
                        break
                    randomness = 0.1 if attempt > 20 else 0.0
                    new_row = generate_row(used_other, current_max, randomness)
                    if new_row is None:
                        continue
                    if max(new_row) < best_max:
                        test_rows = [list(r) for r in rows]
                        test_rows[i] = new_row
                        if is_valid(test_rows):
                            best_row = new_row
                            best_max = max(new_row)
                
                if best_max < max(rows[i]):
                    rows[i] = best_row
                    if scope(rows) < scope([list(r) for r in rows] if False else rows):
                        improved = True
        return rows

    # ---------- Backtracking search for final polish ----------
    def backtracking_polish(rows, time_budget):
        """Bounded backtracking to try to reduce scope by 1 or 2."""
        rows = [list(r) for r in rows]
        current_scope = scope(rows)
        end_time = time.time() + time_budget
        
        # Try to find a row that can be reduced
        for target_scope in range(current_scope - 1, max(current_scope - 5, 100), -1):
            if time.time() > end_time:
                break
            for i in range(n_rows):
                if time.time() > end_time:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                
                # Try to build a row with max <= target_scope
                row, x = [0], 0
                row_diffs = set()
                success = True
                while len(row) < k + 1:
                    found = False
                    for trial_x in range(x + 1, target_scope + 1):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & row_diffs):
                            row.append(trial_x)
                            row_diffs |= new
                            found = True
                            break
                    if not found:
                        success = False
                        break
                    x = row[-1]
                
                if success:
                    test_rows = [list(r) for r in rows]
                    test_rows[i] = row
                    if is_valid(test_rows) and scope(test_rows) < scope(rows):
                        rows = test_rows
                        return rows
        
        return rows

    # Main search loop
    if time_left() > 15:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Annealing phase
    anneal_time = min(60.0, time_left() * 0.5)
    if anneal_time > 10:
        annealed, annealed_scope = anneal(best_rows, anneal_time)
        if annealed_scope < best_scope:
            best_rows = annealed
            best_scope = annealed_scope

    # Polish after annealing
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Backtracking polish
    bt_time = min(30.0, time_left() * 0.4)
    if bt_time > 5:
        bt_rows = backtracking_polish(best_rows, bt_time)
        if scope(bt_rows) < best_scope:
            best_rows = bt_rows
            best_scope = scope(best_rows)

    # Final local polish
    if time_left() > 5:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    return best_rows
