import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    timeout = 114.0  # Leave 6s margin for 120s limit (evaluator kills at 150s, but task says 120s)
    
    def remaining():
        return timeout - (time.time() - start_time)

    # k+1 integers per row, starting at 0.
    # Need to pick k more integers.
    # Total elements per row: k+1.
    # Task: Difference triangle set (4,7) -> n_rows=4, k=7 (length 8).
    
    needed_diffs = k * (k + 1) // 2  # Number of pairwise differences per row

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def scope_of(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0 or any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
                return False
            d = row_diffs_set(r)
            if len(d) != needed_diffs:
                return False
            if d & used:
                return False
            used |= d
        return True

    # ---- 1. Deterministic Greedy (Baseline) ----
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # ---- 2. Randomized Greedy (Diversification) ----
    def rand_greedy(rng, bias, window):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                cands = []
                y = x + 1
                scanned = 0
                # Look for a window of valid candidates
                while len(cands) < window and scanned < 5000:
                    scanned += 1
                    new = {y - a for a in row}
                    if not (new & used):
                        cands.append(y)
                    y += 1
                
                # If no candidates found in window (unlikely for small k), just take the next valid one
                if not cands:
                    while True:
                        new = {y - a for a in row}
                        if not (new & used):
                            cands.append(y)
                            break
                        y += 1
                
                # Choose from candidates
                if rng.random() < bias:
                    chosen = cands[0]  # Smallest
                else:
                    # Biased random: prefer smaller indices
                    idx = int(len(cands) * (rng.random() ** 1.5))
                    idx = min(idx, len(cands) - 1)
                    chosen = cands[idx]
                
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
                x = chosen
            rows.append(row)
        return rows

    # ---- 3. Simulated Annealing / Local Search ----
    # Structural change: Instead of pure DFS backtracking which can get stuck or be slow due to Python overhead,
    # we use a fast local search on a valid solution. This allows exploring the "landscape" of valid solutions
    # to minimize the scope (max entry).
    
    def local_search(start, rng, steps, T0=2.0):
        if not is_valid(start):
            return start
        
        cur = [list(r) for r in start]
        rowd = [row_diffs_set(r) for r in cur]
        cur_used = set()
        for d in rowd:
            cur_used |= d
        
        row_sums = [sum(r) for r in cur]
        row_maxes = [max(r) for r in cur]
        cur_sum = sum(row_sums)
        cur_scope = max(row_maxes)
        
        # Objective: Primary is scope, secondary is sum (to encourage compactness)
        cur_obj = cur_scope * 1000000 + cur_sum
        
        best = [list(r) for r in cur]
        best_obj = cur_obj
        T = T0
        
        for it in range(steps):
            if it % 50 == 0 and remaining() < 10:
                break
            
            # Pick a random row
            r_idx = rng.randrange(n_rows)
            row = cur[r_idx]
            
            # Pick a move type
            move_type = rng.random()
            
            if move_type < 0.40:
                # Compress the maximum element (most likely to reduce scope)
                p = row.index(max(row))
                if p == 0:
                    continue
                lo = row[p - 1] + 1
                hi = row[p] - 1
                if lo > hi:
                    continue
                # Bias towards smaller values
                span = hi - lo + 1
                v = lo + int(span * (rng.random() ** 1.5))
                
            elif move_type < 0.60:
                # Step down max by small amount
                p = row.index(max(row))
                if p == 0:
                    continue
                step = rng.randint(1, 3)
                v = row[p] - step
                if v <= row[p - 1]:
                    continue
                if v > row[p]: # Shouldn't happen
                    continue
                    
            elif move_type < 0.80:
                # Move an interior element (non-max, non-min)
                # This can free up difference space
                if len(row) <= 2:
                    continue
                p = rng.randrange(1, len(row) - 1)
                lo = row[p - 1] + 1
                hi = row[p + 1] - 1
                if lo > hi:
                    continue
                v = rng.randrange(lo, hi + 1)
                
            else:
                # Move any element (including max) to a random valid position
                p = rng.randrange(1, len(row))
                lo = row[p - 1] + 1
                hi = (row[p + 1] - 1) if p + 1 < len(row) else (cur_scope + 10)
                if lo > hi:
                    continue
                v = rng.randrange(lo, hi + 1)
            
            if v == row[p]:
                continue
            
            new_row = row[:]
            new_row[p] = v
            
            # Check strict increase
            if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                continue
            
            nd = row_diffs_set(new_row)
            if len(nd) != needed_diffs:
                continue
            
            # Check collision with other rows
            other_used = cur_used - rowd[r_idx]
            if nd & other_used:
                continue
            
            # Calculate new objective
            new_max_row = max(new_row)
            new_sum_row = sum(new_row)
            
            if new_max_row < row_maxes[r_idx]:
                if row_maxes[r_idx] == cur_scope:
                    # Scope might decrease
                    others_max = max(row_maxes[i] for i in range(n_rows) if i != r_idx) if n_rows > 1 else 0
                    new_scope = max(new_max_row, others_max)
                else:
                    new_scope = cur_scope
            else:
                new_scope = max(cur_scope, new_max_row)
            
            new_sum = cur_sum - row_sums[r_idx] + new_sum_row
            new_obj = new_scope * 1000000 + new_sum
            
            d_obj = new_obj - cur_obj
            
            # Accept if better or with probability exp(-d/T)
            if d_obj <= 0 or rng.random() < math.exp(-d_obj / T):
                cur[r_idx] = new_row
                rowd[r_idx] = nd
                cur_used = other_used | nd
                row_sums[r_idx] = new_sum_row
                row_maxes[r_idx] = new_max_row
                cur_sum = new_sum
                cur_scope = new_scope
                cur_obj = new_obj
                
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [list(x) for x in cur]
            
            # Cool down
            T = max(0.01, T * 0.9998)
        
        return best

    # ---- Main Search Loop ----
    
    # Phase 1: Get a good baseline
    base = greedy()
    best_rows = base
    best_scope = scope_of(base)
    
    rng = random.Random(20240607)
    
    # Phase 2: Randomized Greedy Restarts + Quick Polish
    # This explores diverse starting points.
    while remaining() > 20:
        bias = rng.uniform(0.5, 0.9)
        window = rng.choice([3, 5, 8, 10])
        rows = rand_greedy(rng, bias, window)
        s = scope_of(rows)
        
        if s < best_scope:
            best_scope = s
            best_rows = rows
        
        # Polish promising candidates
        if s < best_scope + 5 and remaining() > 15:
            polished = local_search(rows, rng, 500)
            ps = scope_of(polished)
            if ps < best_scope:
                best_scope = ps
                best_rows = polished
                
    # Phase 3: Deep Annealing from Best
    # Use the best solution found so far as a seed for longer annealing runs.
    for _ in range(5):
        if remaining() < 5:
            break
        
        # Perturb the best solution slightly to escape local minima
        start = [list(r) for r in best_rows]
        # Apply a few random valid moves to perturb
        for _ in range(10):
            r_idx = rng.randrange(n_rows)
            row = start[r_idx]
            p = rng.randrange(1, len(row))
            lo = row[p - 1] + 1
            hi = (row[p + 1] - 1) if p + 1 < len(row) else (best_scope + 5)
            if lo > hi:
                continue
            v = rng.randrange(lo, hi + 1)
            if v == row[p]:
                continue
            new_row = row[:]
            new_row[p] = v
            if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                continue
            # Check validity quickly
            nd = row_diffs_set(new_row)
            if len(nd) != needed_diffs:
                continue
            # Check against other rows in start
            valid = True
            used_other = set()
            for i, r in enumerate(start):
                if i == r_idx:
                    continue
                used_other |= row_diffs_set(r)
            if nd & used_other:
                continue
            start[r_idx] = new_row
        
        polished = local_search(start, rng, 2000, T0=3.0)
        ps = scope_of(polished)
        if ps < best_scope:
            best_scope = ps
            best_rows = polished

    # Final check
    if not is_valid(best_rows):
        # Fallback to greedy if something went wrong
        best_rows = greedy()

    return best_rows

# EVOLVE-BLOCK-END
