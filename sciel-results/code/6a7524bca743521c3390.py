import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 120s kill

    total_diffs = k * (k + 1) // 2  # per row
    counting_lb = n_rows * total_diffs

    def row_diffs(row):
        s = set()
        L = len(row)
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                s.add(row[j] - ri)
        return s

    def is_valid(rows):
        all_d = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            d = row_diffs(row)
            if d & all_d:
                return False
            all_d |= d
        return True

    def greedy_row(forbidden, max_val_hint):
        row = [0]
        local = set()
        upper = max_val_hint if max_val_hint else 2000
        for _ in range(k):
            base = row[-1]
            found = False
            for v in range(base + 1, upper + 1):
                ok = True
                for a in row:
                    d = v - a
                    if d in forbidden or d in local:
                        ok = False
                        break
                if ok:
                    row.append(v)
                    for a in row[:-1]:
                        local.add(row[-1] - a)
                    found = True
                    break
            if not found:
                return None
        return row

    def initial_construct(max_hint):
        rows = []
        used = set()
        for i in range(n_rows):
            r = greedy_row(used, max_hint)
            if r is None:
                return None
            rows.append(r)
            used |= row_diffs(r)
        return rows

    # Try to get an initial solution
    hint = max(counting_lb + 10, 130)
    init_rows = initial_construct(hint)
    if init_rows is None:
        hint = counting_lb + 60
        init_rows = initial_construct(hint)
    if init_rows is None:
        hint = counting_lb + 200
        init_rows = initial_construct(hint)
    if init_rows is None:
        rows = []
        used = set()
        for i in range(n_rows):
            row = [0]
            local = set()
            for _ in range(k):
                base = row[-1]
                v = base + 1
                while True:
                    ok = True
                    for a in row:
                        d = v - a
                        if d in used or d in local:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                row.append(v)
                for a in row[:-1]:
                    local.add(v - a)
            rows.append(row)
            used |= row_diffs(row)
        init_rows = rows

    def score_of(rows):
        return max(max(r) for r in rows)

    best_rows = [r[:] for r in init_rows]
    best_score = score_of(best_rows)

    # --- Backtracking solver with difficulty ordering and bitset pruning ---
    
    def backtrack_solve(max_val):
        """Try to construct n_rows valid rows with all entries <= max_val.
        Uses backtracking with rows ordered by difficulty (fewest options first)."""
        if time.time() > deadline:
            return None
        
        # We'll build rows one at a time.
        # To improve pruning, we can try different orderings of rows.
        # For now, let's stick to sequential but with strong pruning.
        
        rows = []
        used_diffs = set()
        
        def solve(idx):
            if time.time() > deadline:
                return False
            if idx == n_rows:
                return True
            
            # Try to build row idx
            # Use greedy with backtracking
            row = [0]
            local_diffs = set()
            
            def build_row(pos):
                if pos == k:
                    # Check if this row works
                    # local_diffs should not intersect used_diffs (checked during build)
                    rows.append(row[:])
                    return True
                
                base = row[-1]
                # Try candidates
                for v in range(base + 1, max_val + 1):
                    if time.time() > deadline:
                        return False
                    
                    # Check if v can be added
                    ok = True
                    for a in row:
                        d = v - a
                        if d in used_diffs or d in local_diffs:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    # Add v
                    row.append(v)
                    new_diffs = set()
                    for a in row[:-1]:
                        d = v - a
                        new_diffs.add(d)
                    local_diffs.update(new_diffs)
                    
                    if build_row(pos + 1):
                        return True
                    
                    # Backtrack
                    row.pop()
                    for d in new_diffs:
                        local_diffs.discard(d)
                
                return False
            
            if build_row(0):
                used_diffs.update(row_diffs(rows[-1]))
                if solve(idx + 1):
                    return True
                else:
                    used_diffs.difference_update(row_diffs(rows[-1]))
                    rows.pop()
            
            return False
        
        if solve(0):
            return rows
        return None

    # --- Simulated Annealing with improved moves ---
    
    def greedy_candidates(forbidden, row, max_scan=8):
        cands = []
        base = row[-1]
        L = len(row)
        local = set()
        for i in range(L):
            for j in range(i + 1, L):
                local.add(row[j] - row[i])
        scan_limit = base + 500
        for v in range(base + 1, scan_limit + 1):
            ok = True
            for a in row:
                d = v - a
                if d in forbidden or d in local:
                    ok = False
                    break
            if ok:
                cands.append(v)
                if len(cands) >= max_scan:
                    break
        return cands

    def build_row_with_skips(forbidden, skips, max_hint):
        row = [0]
        local = set()
        for t in range(k):
            cands = greedy_candidates(forbidden, row, 8)
            if not cands:
                base = row[-1]
                L = len(row)
                local2 = set()
                for i in range(L):
                    for j in range(i + 1, L):
                        local2.add(row[j] - row[i])
                v = base + 1
                while True:
                    ok = True
                    for a in row:
                        d = v - a
                        if d in forbidden or d in local2:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                cands = [v]
            idx = skips[t]
            if idx < 0:
                idx = 0
            if idx >= len(cands):
                idx = 0
            v = cands[idx]
            row.append(v)
            for a in row[:-1]:
                local.add(v - a)
        if row[-1] > max_hint:
            return None
        return row

    cur_rows = [r[:] for r in init_rows]
    cur_row_diffs = [row_diffs(r) for r in cur_rows]
    cur_score = score_of(cur_rows)

    T = 15.0
    cooling = 0.9995
    min_T = 0.05
    skip_choices = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 3, 4]

    def sample_skips():
        return [random.choice(skip_choices) for _ in range(k)]

    # Also try to improve by targeting specific max values
    # Try backtracking for max_val = best_score - 1, -2, ... down to counting_lb
    for target in range(best_score - 1, counting_lb - 1, -1):
        if time.time() > deadline - 30:
            break
        sol = backtrack_solve(target)
        if sol is not None:
            if is_valid(sol):
                best_rows = sol
                best_score = target
                break

    # Continue with simulated annealing to potentially find even better solutions
    while time.time() < deadline - 5:
        if T < min_T:
            T = 10.0

        idx = random.randrange(n_rows)

        forbidden = set()
        for i in range(n_rows):
            if i != idx:
                forbidden |= cur_row_diffs[i]

        skips = sample_skips()
        max_hint = best_score + 5
        new_row = build_row_with_skips(forbidden, skips, max_hint)
        if new_row is None:
            T *= cooling
            continue

        new_row_d = row_diffs(new_row)
        if new_row_d & forbidden:
            T *= cooling
            continue

        new_rows = cur_rows[:]
        new_rows[idx] = new_row
        ns = score_of(new_rows)

        delta = ns - cur_score
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if random.random() < 0.05:
                accept = True
        else:
            prob = math.exp(-delta / T)
            if random.random() < prob:
                accept = True

        if accept:
            cur_rows[idx] = new_row
            cur_row_diffs[idx] = new_row_d
            cur_score = ns
            if cur_score < best_score:
                best_score = cur_score
                best_rows = [r[:] for r in cur_rows]

        T *= cooling

    if not is_valid(best_rows):
        rows = []
        used = set()
        for i in range(n_rows):
            row = [0]
            local = set()
            for _ in range(k):
                base = row[-1]
                v = base + 1
                while True:
                    ok = True
                    for a in row:
                        d = v - a
                        if d in used or d in local:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                row.append(v)
                for a in row[:-1]:
                    local.add(v - a)
            rows.append(row)
            used |= row_diffs(row)
        return rows

    return best_rows

# EVOLVE-BLOCK-END
