import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0

    def _elapsed():
        return time.time() - start_time

    def _valid(rows):
        all_d = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_d:
                        return False
                    all_d.add(d)
        return True

    def _scope(rows):
        return max(max(r) for r in rows)

    def _greedy_v2(seed_offset=0, max_val=None):
        rng = random.Random(42 + seed_offset)
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            row_diffs = set()
            x = 0
            found_row = False
            while len(row) < k + 1:
                found = False
                attempts = 0
                while not found and attempts < 100:
                    attempts += 1
                    if attempts <= 30:
                        x = row[-1] + rng.randint(1, 20)
                    else:
                        x = row[-1] + 1
                    
                    if max_val is not None and x > max_val:
                        if attempts > 30:
                            break
                        continue
                        
                    conflict = False
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in row_diffs:
                            conflict = True
                            break
                        if (1 << d) & used_mask:
                            conflict = True
                            break
                        new_diffs.append(d)
                    
                    if not conflict:
                        row.append(x)
                        for d in new_diffs:
                            row_diffs.add(d)
                            used_mask |= (1 << d)
                        found = True
                
                if not found:
                    return None
            rows.append(row)
        return rows

    def _backtrack_solve(M, rng):
        used_mask = 0
        rows = []
        
        def _build_row(used_mask, row, row_diffs):
            if len(row) == k + 1:
                return True
            
            last = row[-1]
            candidates = []
            # Generate candidates: small steps and random
            for d in range(1, min(15, M - last + 1)):
                candidates.append(last + d)
            for _ in range(15):
                candidates.append(last + rng.randint(1, max(1, M - last)))
            
            candidates = list(set(candidates))
            rng.shuffle(candidates)
            
            for x in candidates:
                if x > M:
                    continue
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in row_diffs:
                        valid = False
                        break
                    if (1 << d) & used_mask:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                row.append(x)
                new_used_mask = used_mask
                for d in new_diffs:
                    new_used_mask |= (1 << d)
                    row_diffs.add(d)
                
                if _build_row(new_used_mask, row, row_diffs):
                    return True
                
                row.pop()
                for d in new_diffs:
                    row_diffs.remove(d)
                    
            return False

        for i in range(n_rows):
            row = [0]
            row_diffs = set()
            if not _build_row(used_mask, row, row_diffs):
                return None
            rows.append(row)
            for d in row_diffs:
                used_mask |= (1 << d)
        
        return rows

    # Phase 1: Greedy Init
    best_rows = None
    best_scope = float('inf')
    
    for seed in range(20):
        if _elapsed() > deadline - 20: break
        rows = _greedy_v2(seed)
        if rows:
            scope = _scope(rows)
            if scope < best_scope:
                best_rows = rows
                best_scope = scope

    if best_rows is None:
        return [[0, 1, 2, 3, 4, 5]]

    # Phase 2: Local Search Shrink
    rng = random.Random(123)
    improved = True
    while improved and _elapsed() < deadline - 10:
        improved = False
        current_scope = _scope(best_rows)
        for ri in range(len(best_rows)):
            row = best_rows[ri]
            for pos in range(k, 0, -1):
                old_val = row[pos]
                if old_val < current_scope: continue
                low = row[pos-1] + 1
                high = old_val - 1
                if pos < k:
                    high = min(high, row[pos+1] - 1)
                
                if low > high: continue
                
                for new_val in range(low, high + 1):
                    test_rows = [list(r) for r in best_rows]
                    test_rows[ri][pos] = new_val
                    if _valid(test_rows):
                        best_rows = test_rows
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Phase 3: Bitset Backtracking
    current_M = _scope(best_rows)
    for target_M in range(current_M - 1, max(current_M - 10, 100) - 1, -1):
        if _elapsed() > deadline - 5:
            break
        found = False
        for attempt in range(5):
            if _elapsed() > deadline - 5:
                break
            rng_bt = random.Random(int(_elapsed() * 1000) + attempt)
            rows = _backtrack_solve(target_M, rng_bt)
            if rows:
                best_rows = rows
                best_scope = target_M
                found = True
                break
        if found:
            current_M = target_M
            if _elapsed() < deadline - 10:
                continue
            else:
                break

    if not _valid(best_rows):
        best_rows = _greedy_v2(0)
        if best_rows is None:
            return [[0, 1, 2, 3, 4, 5]]

    return best_rows
