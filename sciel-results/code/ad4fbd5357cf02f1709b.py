import time
import random
import sys

def construct(n_rows, k):
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s wall-clock

    def row_diffs(row):
        out = set()
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - a)
        return out

    def validate(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    def max_entry(rows):
        return max(row[-1] for row in rows)

    def greedy_row(used_set, k, cap=None):
        """Smallest-next-mark row avoiding used_set."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set = used_set | new
        return row

    def dfs_row_bidir(row, used_set, cap, row_deadline):
        """DFS with bidirectional pruning."""
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k + 1:
            return row[:]
        last = row[-1]
        needed = (k + 1) - len(row)
        min_next = last + 1
        max_next = cap - (needed - 1)
        if min_next > max_next:
            return None
        if cap - last < needed:
            return None
        for x in range(min_next, max_next + 1):
            if time.time() > row_deadline or time.time() > deadline:
                return None
            new = {x - a for a in row}
            if new & used_set:
                continue
            res = dfs_row_bidir(row + [x], used_set | new, cap, row_deadline)
            if res is not None:
                return res
        return None

    def randomized_greedy(n_rows, k, rng, max_backtracks=2):
        """Build all rows using randomized greedy with optional backtracking."""
        used = set()
        rows = []
        
        for r in range(n_rows):
            row = [0]
            bt = max_backtracks if r < n_rows // 2 else 0
            
            def dfs_row(depth, current_row, used_local, bt_rem):
                if depth == k + 1:
                    return True, current_row
                last = current_row[-1]
                max_candidate = last + 80
                candidates = list(range(last + 1, max_candidate + 1))
                rng.shuffle(candidates)
                candidates.sort(key=lambda x: x * 0.7 + rng.random() * 20)
                
                for x in candidates:
                    new_diffs = {x - a for a in current_row}
                    if new_diffs & used_local:
                        continue
                    new_row = current_row + [x]
                    new_used = used_local | new_diffs
                    
                    if depth == k:
                        return True, new_row
                    
                    success, result = dfs_row(depth + 1, new_row, new_used, bt_rem - 1)
                    if success:
                        return True, result
                    if bt_rem <= 0:
                        break
                return False, None
            
            success, result = dfs_row(1, row, used, bt)
            if not success:
                # Fallback to pure greedy
                row = [0]
                while len(row) < k + 1:
                    x = row[-1] + 1
                    while True:
                        new = {x - a for a in row}
                        if not new & used:
                            row.append(x)
                            used |= new
                            break
                        x += 1
                result = row
            
            rows.append(result)
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    used.add(result[j] - result[i])
        
        return rows

    def try_compression(rows, used_diffs, k, rng, max_iters=100):
        """Try to reduce the maximum entry by locally modifying rows."""
        rows = [list(r) for r in rows]
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Find the row with the largest entry
            max_row_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
            row = rows[max_row_idx]
            
            if row[-1] <= 0:
                break
            
            old_last = row[-1]
            row_diffs_set = row_diffs(row)
            other_diffs = used_diffs - row_diffs_set
            
            # Try smaller values for the last element
            found = False
            for new_last in range(row[-2] + 1, old_last):
                new_row = row[:-1] + [new_last]
                new_diffs = row_diffs(new_row)
                
                if not (new_diffs & other_diffs):
                    rows[max_row_idx] = new_row
                    used_diffs = other_diffs | new_diffs
                    found = True
                    break
            
            if not found:
                # Try modifying other elements
                if len(row) >= 3:
                    improved = False
                    for pos in range(len(row) - 2, -1, -1):
                        if time.time() > deadline - 5:
                            break
                        old_val = row[pos]
                        lower_bound = row[pos - 1] if pos > 0 else -1
                        for new_val in range(old_val - 1, lower_bound, -1):
                            test_row = row[:]
                            test_row[pos] = new_val
                            if all(test_row[i] < test_row[i + 1] for i in range(len(test_row) - 1)):
                                test_diffs = row_diffs(test_row)
                                if not (test_diffs & other_diffs):
                                    rows[max_row_idx] = test_row
                                    used_diffs = other_diffs | test_diffs
                                    improved = True
                                    break
                        if improved:
                            break
        
        return rows, used_diffs

    # ---- Phase 1: greedy baseline (always valid) ----
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        rows.append(row)
        used |= row_diffs(row)
    
    best_rows = [list(r) for r in rows]
    best_max = max_entry(rows)

    # ---- Phase 2: per-row DFS refinement during construction ----
    for r in range(n_rows):
        if time.time() > deadline - 10:
            break
        cur_last = rows[r][-1]
        used_without = set(used)
        used_without.difference_update(row_diffs(rows[r]))
        for cap in range(cur_last - 1, max(cur_last - 20, 0), -1):
            if time.time() > deadline - 10:
                break
            row_deadline = min(deadline - 10, time.time() + 3)
            cand = dfs_row_bidir([0], used_without, cap, row_deadline)
            if cand is not None:
                used.difference_update(row_diffs(rows[r]))
                rows[r] = cand
                used |= row_diffs(cand)
                break

    if validate(rows):
        m = max_entry(rows)
        if m < best_max:
            best_max = m
            best_rows = [list(r) for r in rows]

    # ---- Phase 3: bounded iterative improvement (replace rows with smaller ones) ----
    order = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for r in order:
            if time.time() > deadline - 5:
                break
            cur_last = rows[r][-1]
            best = None
            for cap in range(cur_last - 1, max(cur_last - 40, 0), -1):
                if time.time() > deadline - 5:
                    break
                used_without = set(used)
                used_without.difference_update(row_diffs(rows[r]))
                row_deadline = min(deadline - 5, time.time() + 4)
                cand = dfs_row_bidir([0], used_without, cap, row_deadline)
                if cand is not None:
                    best = cand
                    break
            if best is not None and best[-1] < cur_last:
                used.difference_update(row_diffs(rows[r]))
                rows[r] = best
                used |= row_diffs(best)
                improved = True

    if validate(rows):
        m = max_entry(rows)
        if m < best_max:
            best_max = m
            best_rows = [list(r) for r in rows]

    # ---- Phase 4: local search to reduce max entry within a row ----
    if time.time() < deadline - 10:
        for r in range(n_rows):
            if time.time() > deadline - 10:
                break
            row = rows[r]
            cur_max = row[-1]
            used_without = set(used)
            used_without.difference_update(row_diffs(row))
            
            improved_row = None
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                cand_row = row[:]
                for pos in range(len(cand_row) - 2, -1, -1):
                    if time.time() > deadline - 10:
                        break
                    old_val = cand_row[pos]
                    lower_bound = cand_row[pos - 1] if pos > 0 else -1
                    for new_val in range(old_val - 1, lower_bound, -1):
                        if time.time() > deadline - 10:
                            break
                        test_row = cand_row[:]
                        test_row[pos] = new_val
                        test_diffs = row_diffs(test_row)
                        if not (test_diffs & used_without):
                            if all(test_row[i] < test_row[i + 1] for i in range(len(test_row) - 1)):
                                cand_row = test_row
                                break
                if cand_row[-1] < cur_max:
                    improved_row = cand_row
                    break
            
            if improved_row is not None and improved_row[-1] < cur_max:
                used.difference_update(row_diffs(rows[r]))
                rows[r] = improved_row
                used |= row_diffs(improved_row)

    if validate(rows):
        m = max_entry(rows)
        if m < best_max:
            best_max = m
            best_rows = [list(r) for r in rows]

    # ---- Phase 5: randomized restarts with compression ----
    if time.time() < deadline - 15:
        rng = random.Random(42)
        for attempt in range(10):
            if time.time() > deadline - 15:
                break
            
            # Randomized greedy construction
            rand_rows = randomized_greedy(n_rows, k, rng, max_backtracks=2)
            
            if not validate(rand_rows):
                continue
            
            rand_used = set()
            for row in rand_rows:
                rand_used |= row_diffs(row)
            
            # Apply compression
            rand_rows, rand_used = try_compression(rand_rows, rand_used, k, rng, max_iters=50)
            
            if validate(rand_rows):
                m = max_entry(rand_rows)
                if m < best_max:
                    best_max = m
                    best_rows = [list(r) for r in rand_rows]

    return best_rows
