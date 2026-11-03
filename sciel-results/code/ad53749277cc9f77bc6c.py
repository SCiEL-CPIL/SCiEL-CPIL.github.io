import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 140.0  # safety margin before 150s kill
    row_len = k + 1
    n_diffs_per_row = row_len * (row_len - 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    # Bitmask helpers
    def diffs_of_row(row):
        """Return set of positive differences in a row."""
        ds = set()
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                ds.add(row[j] - ri)
        return ds

    def is_valid_full(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != row_len:
                return False
            if row[0] != 0:
                return False
            if any(row[j] <= row[j-1] for j in range(1, len(row))):
                return False
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def scope_of(rows):
        return max(r[-1] for r in rows)

    # Phase 1: Fast greedy to get initial solution
    def greedy_construct(max_scope, seed=0):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            local_diffs = set()
            for pos in range(1, row_len):
                found = False
                # Try candidates in increasing order with some randomization
                candidates = list(range(row[-1] + 1, max_scope + 1))
                if seed > 0 and pos > 1:
                    # Randomize slightly
                    rng.shuffle(candidates)
                    # But keep small values somewhat prioritized
                    candidates.sort(key=lambda x: (x > row[-1] + 3, x))
                for t in candidates:
                    new_diffs = set()
                    for a in row:
                        new_diffs.add(t - a)
                    if new_diffs & used or new_diffs & local_diffs:
                        continue
                    row.append(t)
                    local_diffs |= new_diffs
                    found = True
                    break
                if not found:
                    return None
            rows.append(row)
            used |= diffs_of_row(row)
        return rows

    # Get initial solution
    best_rows = None
    best_scope = float('inf')
    
    for seed in range(20):
        if time.time() > DEADLINE - 30:
            break
        rows = greedy_construct(200, seed)
        if rows is not None:
            s = scope_of(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: very conservative greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(1, row_len):
                v = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (v - a) in used:
                            ok = False
                            break
                    if ok:
                        row.append(v)
                        for a in row[:-1]:
                            used.add(v - a)
                        break
                    v += 1
            rows.append(row)
        return rows

    # Phase 2: Structured local search with row-swap and compression
    # Key structural change: treat the set of rows as a hypergraph and do
    # row-level moves that preserve validity, plus "scope reduction" moves
    # that try to push the maximum element down.
    
    def build_diff_map(rows):
        """Map each difference to the row index that uses it."""
        dmap = {}
        for ri, row in enumerate(rows):
            for d in diffs_of_row(row):
                if d in dmap:
                    return None  # collision
                dmap[d] = ri
        return dmap

    def row_diffs_set(row):
        return diffs_of_row(row)

    # Simulated annealing on row assignments
    if time.time() < DEADLINE - 5:
        cur = [list(r) for r in best_rows]
        cur_scope = scope_of(cur)
        best = [list(r) for r in cur]
        best_scope = cur_scope
        rng = random.Random(123)
        
        T = 3.0
        cooling = 0.9997
        min_T = 0.01
        iters = 0
        max_iters = 200000
        
        while time.time() < DEADLINE - 3 and T > min_T and iters < max_iters:
            iters += 1
            move_type = rng.random()
            
            if move_type < 0.4:
                # Rebuild one row greedily with tighter scope
                i = rng.randint(0, n_rows - 1)
                old_row = list(cur[i])
                old_max = cur_scope
                # Build used set without row i
                used_others = set()
                for ri in range(n_rows):
                    if ri != i:
                        used_others |= diffs_of_row(cur[ri])
                
                # Try to build a row with smaller max
                target_max = old_max - 1
                new_row = None
                for attempt in range(3):
                    row = [0]
                    local = set()
                    ok = True
                    for pos in range(1, row_len):
                        found = False
                        for t in range(row[-1] + 1, target_max + 1):
                            nd = set()
                            for a in row:
                                nd.add(t - a)
                            if nd & used_others or nd & local:
                                continue
                            row.append(t)
                            local |= nd
                            found = True
                            break
                        if not found:
                            ok = False
                            break
                    if ok and row[-1] < old_row[-1]:
                        new_row = row
                        break
                
                if new_row is not None:
                    new_scope = scope_of(cur[:i] + [new_row] + cur[i+1:])
                    delta = new_scope - cur_scope
                    if delta < 0 or rng.random() < math.exp(-delta / max(T, min_T)):
                        cur[i] = new_row
                        cur_scope = new_scope
                        if cur_scope < best_scope:
                            best_scope = cur_scope
                            best = [list(r) for r in cur]
            
            elif move_type < 0.7:
                # Swap two rows' assignments via rebuild
                i, j = rng.sample(range(n_rows), 2)
                old_i = list(cur[i])
                old_j = list(cur[j])
                
                used_others = set()
                for ri in range(n_rows):
                    if ri != i and ri != j:
                        used_others |= diffs_of_row(cur[ri])
                
                new_i = None
                new_j = None
                # Rebuild i
                row = [0]
                local = set()
                ok = True
                for pos in range(1, row_len):
                    found = False
                    for t in range(row[-1] + 1, cur_scope + 1):
                        nd = set()
                        for a in row:
                            nd.add(t - a)
                        if nd & used_others or nd & local:
                            continue
                        row.append(t)
                        local |= nd
                        found = True
                        break
                    if not found:
                        ok = False
                        break
                if ok:
                    new_i = row
                    used_others |= diffs_of_row(new_i)
                
                # Rebuild j
                if new_i is not None:
                    row = [0]
                    local = set()
                    ok = True
                    for pos in range(1, row_len):
                        found = False
                        for t in range(row[-1] + 1, cur_scope + 1):
                            nd = set()
                            for a in row:
                                nd.add(t - a)
                            if nd & used_others or nd & local:
                                continue
                            row.append(t)
                            local |= nd
                            found = True
                            break
                        if not found:
                            ok = False
                            break
                    if ok:
                        new_j = row
                
                if new_i is not None and new_j is not None:
                    new_scope = max(new_i[-1], new_j[-1], 
                                   max(r[-1] for ri, r in enumerate(cur) if ri != i and ri != j))
                    delta = new_scope - cur_scope
                    if delta < 0 or rng.random() < math.exp(-delta / max(T, min_T)):
                        cur[i] = new_i
                        cur[j] = new_j
                        cur_scope = new_scope
                        if cur_scope < best_scope:
                            best_scope = cur_scope
                            best = [list(r) for r in cur]
            
            else:
                # Single element nudge
                i = rng.randint(0, n_rows - 1)
                j = rng.randint(1, row_len - 1)
                old_row = list(cur[i])
                old_val = old_row[j]
                
                used_others = set()
                for ri in range(n_rows):
                    if ri != i:
                        used_others |= diffs_of_row(cur[ri])
                
                improved = False
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    if j > 0 and new_val <= old_row[j-1]:
                        continue
                    if j < row_len - 1 and new_val >= old_row[j+1]:
                        continue
                    new_row = list(old_row)
                    new_row[j] = new_val
                    nd = diffs_of_row(new_row)
                    if nd & used_others:
                        continue
                    # Check internal validity
                    if len(nd) != n_diffs_per_row:
                        continue
                    new_scope = max(new_row[-1], max(r[-1] for ri, r in enumerate(cur) if ri != i))
                    delta_s = new_scope - cur_scope
                    if delta_s < 0 or rng.random() < math.exp(-delta_s / max(T, min_T)):
                        cur[i] = new_row
                        cur_scope = new_scope
                        if cur_scope < best_scope:
                            best_scope = cur_scope
                            best = [list(r) for r in cur]
                        improved = True
                        break
            
            T *= cooling
    
    # Phase 3: Final polish - try to reduce scope by 1, 2, 3
    if time.time() < DEADLINE - 1:
        for _ in range(50):
            if time.time() > DEADLINE - 1:
                break
            improved = False
            for i in range(n_rows):
                if best[i][-1] < best_scope:
                    continue
                old_row = list(best[i])
                used_others = set()
                for ri in range(n_rows):
                    if ri != i:
                        used_others |= diffs_of_row(best[ri])
                
                # Try to find row with max < old_row[-1]
                for target in range(old_row[-1] - 1, old_row[-1] - 5, -1):
                    row = [0]
                    local = set()
                    ok = True
                    for pos in range(1, row_len):
                        found = False
                        for t in range(row[-1] + 1, target + 1):
                            nd = set()
                            for a in row:
                                nd.add(t - a)
                            if nd & used_others or nd & local:
                                continue
                            row.append(t)
                            local |= nd
                            found = True
                            break
                        if not found:
                            ok = False
                            break
                    if ok:
                        best[i] = row
                        best_scope = scope_of(best)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    # Final validation
    if not is_valid_full(best):
        # Fallback to greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(1, row_len):
                v = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (v - a) in used:
                            ok = False
                            break
                    if ok:
                        row.append(v)
                        for a in row[:-1]:
                            used.add(v - a)
                        break
                    v += 1
            rows.append(row)
        return rows

    return best
