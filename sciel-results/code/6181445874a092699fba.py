"""Difference triangle set (4,7) - difference-first assignment + backtracking + ILS."""
import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140

    def time_left():
        return deadline - time.time()

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                s.add(row[j] - row[i])
        return s

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_set(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(r) for r in rows)

    # Greedy baseline
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

    # Backtracking on difference assignment
    # Each row needs 28 differences that are consistent with an 8-element set starting at 0
    # The 28 differences for a row with prefix sums p1..p7 are:
    # p1, p2, ..., p7, p2-p1, p3-p1, ..., p7-p1, p3-p2, ..., p7-p2, ..., p7-p6
    # Actually, differences are a_j - a_i for i<j, where a_0=0, a_j = p1+...+pj
    # So differences are: a_j (j=1..7) and a_j - a_i (i<j, j>=2)
    
    def backtrack_search(target_scope, time_limit, seed=0):
        random.seed(seed)
        start = time.time()
        
        # We assign differences 1..target_scope to rows
        # Each row gets exactly 28 differences
        # Total needed: 4 * 28 = 112
        # We need to pick 112 differences from 1..target_scope and partition into 4 groups of 28
        # such that each group is "realizable" (comes from a valid 8-element set)
        
        # Strategy: greedily assign smallest available differences, but use backtracking
        # when a row's assigned differences can't form a valid set
        
        rows = [None] * n_rows
        row_diffs = [set() for _ in range(n_rows)]
        row_entries = [[0] for _ in range(n_rows)]  # current entries being built
        used = set()
        
        # We'll build rows one at a time
        # For each row, we need to find 7 more entries such that all 28 differences are unused
        # and max entry <= target_scope
        
        def build_row(row_idx, time_budget):
            """Build a single row with max entry <= target_scope using backtracking."""
            row = [0]
            row_diffs_local = set()
            
            def solve(pos):
                if time.time() > start + time_budget:
                    return False
                if pos == k + 1:
                    # Check all differences are valid
                    for d in row_diffs_local:
                        if d in used or d > target_scope:
                            return False
                    return True
                
                prev = row[-1]
                lower = prev + 1
                upper = target_scope
                
                # Try values in increasing order with some randomization
                candidates = list(range(lower, upper + 1))
                if len(candidates) > 50:
                    # Sample a subset, biased towards smaller
                    sample_size = min(50, len(candidates))
                    # Weight towards smaller values
                    weights = [1.0 / (i + 1) for i in range(len(candidates))]
                    total_w = sum(weights)
                    weights = [w / total_w for w in weights]
                    candidates = [candidates[i] for i in random.choices(range(len(candidates)), weights=weights, k=sample_size)]
                    candidates.sort()
                
                for val in candidates:
                    new_diffs = {val - a for a in row}
                    if any(d in used for d in new_diffs):
                        continue
                    if any(d in row_diffs_local for d in new_diffs):
                        continue
                    if any(d > target_scope for d in new_diffs):
                        continue
                    
                    row.append(val)
                    row_diffs_local |= new_diffs
                    
                    if solve(pos + 1):
                        return True
                    
                    row.pop()
                    row_diffs_local -= new_diffs
                
                return False
            
            if solve(1):
                # Commit this row
                for d in row_diffs_local:
                    used.add(d)
                return row
            return None
        
        for r in range(n_rows):
            if time.time() > start + time_limit:
                return None
            # Allocate more time to later rows (they have fewer available differences)
            remaining_time = time_limit - (time.time() - start)
            row_budget = remaining_time / (n_rows - r)
            row = build_row(r, row_budget)
            if row is None:
                return None
            rows[r] = row
        
        return rows

    # Phase 1: Iterative descent with backtracking
    current_target = best_scope - 1
    seed_counter = 0
    consecutive_failures = 0
    
    while time_left() > 40 and current_target > 108:
        if consecutive_failures > 4:
            break
        
        time_budget = min(5.0, time_left() - 40)
        if time_budget < 1.0:
            break
        
        found = backtrack_search(current_target, time_budget, seed=seed_counter)
        seed_counter += 1
        
        if found:
            if is_valid(found):
                best_rows = found
                best_scope = current_target
                consecutive_failures = 0
                current_target -= 1
            else:
                consecutive_failures += 1
                current_target -= 1
        else:
            consecutive_failures += 1
            if consecutive_failures == 1:
                # Retry once
                found = backtrack_search(current_target, time_budget, seed=seed_counter)
                seed_counter += 1
                if found and is_valid(found):
                    best_rows = found
                    best_scope = current_target
                    consecutive_failures = 0
                    current_target -= 1
                else:
                    consecutive_failures += 1
                    current_target -= 1
            else:
                current_target -= 1

    # Local polish: try to reduce individual rows
    def try_reduce_row(rows, row_idx, max_attempts=100):
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max
        
        used_other = set()
        for j in range(n_rows):
            if j != row_idx:
                used_other |= row_diffs_set(rows[j])
        
        for attempt in range(max_attempts):
            if time_left() < 3:
                break
            row, x = [0], 0
            row_diffs_local = set()
            jitter = 0.1 if attempt > 10 else 0.0
            success = True
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + 100):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & row_diffs_local):
                        if jitter > 0 and random.random() < jitter and trial_x < 200:
                            continue
                        row.append(trial_x)
                        row_diffs_local |= new
                        found = True
                        break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) < best_max:
                best_row = list(row)
                best_max = max(row)
        
        return best_row, best_max

    def local_polish(rows):
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 5
        passes = 0
        while improved and time_left() > 8 and passes < max_passes:
            improved = False
            passes += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 8:
                    break
                new_row, new_max = try_reduce_row(rows, i, max_attempts=60)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    if scope(rows) < old_scope:
                        improved = True
        return rows

    # Phase 2: Polish
    if time_left() > 15:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Phase 3: ILS
    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            row, x = [0], 0
            row_diffs_local = set()
            success = False
            for attempt in range(20):
                row, x = [0], 0
                row_diffs_local = set()
                valid = True
                while len(row) < k + 1:
                    found = False
                    for trial_x in range(x + 1, x + 150):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & row_diffs_local):
                            if random.random() < 0.4 and trial_x < 300:
                                continue
                            row.append(trial_x)
                            row_diffs_local |= new
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                    x = row[-1]
                if valid:
                    rows[idx] = row
                    success = True
                    break
            if not success:
                # Fallback greedy
                row, x = [0], 0
                row_diffs_local = set()
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used_other) and not (new & row_diffs_local):
                        row.append(x)
                        row_diffs_local |= new
                rows[idx] = row
        return rows

    while time_left() > 25:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        if time_left() < 20:
            break
        
        kicked = kick(best_rows)
        polished_kicked = local_polish(kicked)
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Final polish
    if time_left() > 5:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
