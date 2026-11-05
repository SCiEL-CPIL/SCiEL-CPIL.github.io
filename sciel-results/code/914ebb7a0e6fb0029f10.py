import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

    def get_max_entry(rows):
        return max(max(row) for row in rows)

    def get_row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_row(row, global_used_set):
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in row_diffs:
                    return False, set()
                row_diffs.add(d)
        if row_diffs & global_used_set:
            return False, set()
        return True, row_diffs

    def greedy_row(max_val, global_used_set):
        row = [0]
        current_diffs = set()
        
        while len(row) < k + 1:
            last = row[-1]
            found = False
            for v in range(last + 1, max_val + 1):
                conflict = False
                new_diffs = set()
                for u in row:
                    d = v - u
                    if d in current_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                if new_diffs & global_used_set:
                    continue
                
                row.append(v)
                current_diffs |= new_diffs
                found = True
                break
            
            if not found:
                return None
        return row

    def random_row(max_val, global_used_set, attempts=1):
        for _ in range(attempts):
            row = [0]
            current_diffs = set()
            success = True
            for _ in range(k):
                last = row[-1]
                candidates = list(range(last + 1, max_val + 1))
                random.shuffle(candidates)
                found = False
                for v in candidates:
                    conflict = False
                    new_diffs = set()
                    for u in row:
                        d = v - u
                        if d in current_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    if conflict:
                        continue
                    if new_diffs & global_used_set:
                        continue
                    row.append(v)
                    current_diffs |= new_diffs
                    found = True
                    break
                if not found:
                    success = False
                    break
            if success:
                valid, _ = is_valid_row(row, global_used_set)
                if valid:
                    return row
        return None

    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # 1. Initial Solution: Greedy with large scope
    init_scope = 300
    rows = []
    global_used = set()
    for i in range(n_rows):
        row = greedy_row(init_scope, global_used)
        if row is None:
            row = greedy_row(1000, global_used)
        if row is None:
            # Fallback: very sparse
            row = [0] + [i * (k+1) + 1 for i in range(1, k+1)]
            # Ensure distinct diffs within row
            # Actually, just use a simple valid row
            row = [0, 1, 3, 7, 15, 31, 63, 127, 255][:k+1]
            # This might not be valid if k is large, but it's a fallback
            # Better fallback:
            row = [0]
            used_local = set()
            for j in range(1, k+1):
                # Pick a number that doesn't conflict
                # Simple strategy: multiply by a large prime-ish number
                # Actually, let's just use the greedy with a very high limit
                pass
            # Let's just use a simple construction that is always valid:
            # Row i: 0, M*i, M*i+1, ... no, that doesn't work for diffs.
            # Standard valid row: 0, 1, 3, 7, 15, ... (2^j - 1)
            # Diffs are 2^j - 2^i = 2^i(2^(j-i)-1). These are distinct.
            # Max entry is 2^k - 1.
            row = [2**j - 1 for j in range(k+1)]
        
        rows.append(row)
        global_used |= get_row_diffs_set(row)
    
    best_rows = [list(r) for r in rows]
    best_max = get_max_entry(best_rows)
    current_rows = [list(r) for r in rows]
    current_max = best_max

    # 2. Optimization: Simulated Annealing / Local Search
    # We try to reduce the max entry by modifying rows.
    
    # Helper to get all diffs except for a specific row
    def get_all_diffs_except(rows, exclude_idx):
        all_diffs = set()
        for i in range(len(rows)):
            if i == exclude_idx:
                continue
            all_diffs |= get_row_diffs_set(rows[i])
        return all_diffs

    # Local Search Step: Try to replace a row with a "tighter" one
    def try_improve_row(rows, idx, current_max, time_limit):
        """
        Try to find a new row for index idx that has max <= current_max 
        and is valid against other rows. 
        We specifically try to find one with max < rows[idx][-1] if possible,
        or just any valid one to allow reshuffling.
        """
        others_diffs = get_all_diffs_except(rows, idx)
        old_row_max = rows[idx][-1]
        
        # Try to find a row with max < old_row_max
        target_max = old_row_max - 1
        if target_max < k:
            return None
            
        # Try a few random rows
        for _ in range(20):
            if time.time() > time_limit:
                return None
            new_row = random_row(target_max, others_diffs, attempts=1)
            if new_row is not None:
                return new_row
        
        # If we couldn't shrink it, maybe we can just re-randomize it within the same max?
        # This helps escape local minima where the row is "stuck" but a different valid row of same size exists.
        # But we only accept if it allows further improvements elsewhere.
        # For now, let's just return None if we can't shrink.
        return None

    # Main Loop
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
        # Pick a random row to modify
        row_idx = random.randint(0, n_rows - 1)
        
        new_row = try_improve_row(current_rows, row_idx, current_max, deadline - 5.0)
        
        if new_row is not None:
            # Check if this actually reduces the global max
            old_global_max = get_max_entry(current_rows)
            current_rows[row_idx] = new_row
            new_global_max = get_max_entry(current_rows)
            
            if new_global_max < best_max:
                best_max = new_global_max
                best_rows = [list(r) for r in current_rows]
                no_improve_count = 0
            else:
                # Even if global max didn't drop, the row changed. 
                # This might unlock further moves. 
                # But we only track "improvement" as global max drop.
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # Occasionally, try to "kick" a random row to a completely new random valid row
        # with a slightly larger max, to escape local optima?
        # This is risky. Let's stick to shrinking for now.

    # Final Validation
    if not validate(best_rows):
        # Fallback
        rows_fallback = []
        gu = set()
        for i in range(n_rows):
            r = greedy_row(1000, gu)
            if r is None:
                r = [2**j - 1 for j in range(k+1)]
            rows_fallback.append(r)
            gu |= get_row_diffs_set(r)
        best_rows = rows_fallback

    return best_rows
