import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0

    total_diffs = n_rows * (k * (k + 1) // 2)
    lb = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2.0))
    upper = lb + 250

    def time_up():
        return time.time() > deadline

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_baseline(seed=0, max_val=None):
        if max_val is None:
            max_val = upper
        rng = random.Random(seed)
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                best = None
                for x in range(row[-1] + 1, max_val + 1):
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        best = x
                        break
                if best is None:
                    return None
                row.append(best)
                for a in row[:-1]:
                    used.add(best - a)
            rows.append(row)
        return rows

    best_rows = None
    best_max = float('inf')

    def record(rows):
        nonlocal best_rows, best_max
        m = max(max(row) for row in rows)
        if m < best_max:
            best_max = m
            best_rows = [row[:] for row in rows]

    # 1. Initial Greedy Baseline
    for s in range(10):
        if time_up():
            break
        g = greedy_baseline(s)
        if g is not None:
            record(g)

    if best_rows is None:
        return [[0, 1, 2, 3, 4, 5, 6, 7]] if n_rows == 1 else None

    # 2. Simulated Annealing
    # State: rows (list of lists), used_counts (dict diff -> count), diff_to_rows (dict diff -> set of row_idx)
    # We use a fast delta calculation.
    
    def build_state(rows):
        diff_to_rows = {}
        used_counts = {}
        in_row = [set(row) for row in rows]
        for r in range(n_rows):
            for i in range(len(rows[r])):
                for j in range(i + 1, len(rows[r])):
                    d = rows[r][j] - rows[r][i]
                    used_counts[d] = used_counts.get(d, 0) + 1
                    if d not in diff_to_rows:
                        diff_to_rows[d] = set()
                    diff_to_rows[d].add(r)
        return used_counts, diff_to_rows, in_row

    def delta_add(row_idx, x, row_vals, used_counts):
        # Returns change in total collisions (sum c(d)^2) when adding x
        # Cost function: sum_{d} c(d)^2. 
        # Change: (c+1)^2 - c^2 = 2c + 1
        # Total delta = sum (2c_d + 1) for new diffs d
        # Wait, the validity condition is c(d) == 1 for all d.
        # If we use cost = sum (c(d)-1)^2 for all d involved?
        # Simpler: Minimize sum c(d)^2. Validity is sum c(d) = total_diffs AND min c(d) >= 1?
        # No, validity is c(d) <= 1 for all d.
        # If we minimize sum c(d)^2, we encourage spreading differences.
        # If c(d)=1, cost 1. If c(d)=2, cost 4.
        # We want to reach sum c(d) = total_diffs with all c(d)=1 (cost = total_diffs).
        # Start from valid state (cost = total_diffs).
        # If we add a duplicate diff, c becomes 2, cost increases by 3.
        # So minimizing sum c(d)^2 is a good heuristic.
        
        delta = 0
        for a in row_vals:
            if a >= x:
                continue
            d = x - a
            c = used_counts.get(d, 0)
            # Adding x increases count of d by 1.
            # New cost contribution for d: (c+1)^2. Old: c^2.
            # Delta for this d: 2c + 1.
            delta += 2 * c + 1
        return delta

    def delta_remove(row_idx, x, row_vals, used_counts):
        delta = 0
        for a in row_vals:
            if a >= x:
                continue
            d = x - a
            c = used_counts.get(d, 0)
            # Removing x decreases count of d by 1.
            # New cost: (c-1)^2. Old: c^2.
            # Delta: (c-1)^2 - c^2 = c^2 - 2c + 1 - c^2 = 1 - 2c.
            delta += 1 - 2 * c
        return delta

    def apply_move(row_idx, x, sign, rows, used_counts, diff_to_rows, in_row):
        # sign: +1 add, -1 remove
        row_vals = sorted(in_row[row_idx])
        # Update global structures
        for a in row_vals:
            if a >= x:
                continue
            d = x - a
            old_c = used_counts.get(d, 0)
            new_c = old_c + sign
            if new_c <= 0:
                used_counts.pop(d, None)
                # Remove from diff_to_rows?
                # diff_to_rows[d] contains row_idx if any element in row_idx creates diff d.
                # We need to know if row_idx still has other diffs.
                # This is expensive to track perfectly without re-scanning.
                # Approximation: just track counts for cost. For validity check, we do it at the end.
                # For SA, we just need cost.
            else:
                used_counts[d] = new_c
            
            # Update diff_to_rows for tracking which rows have which diffs (used for pruning/moves?)
            # Actually, for delta calculation, we only need counts.
            # But to propose moves, we might want to know which diffs are "hot".
            
        in_row[row_idx].add(x) if sign == 1 else in_row[row_idx].discard(x)
        rows[row_idx] = sorted(in_row[row_idx])
        return rows, used_counts, diff_to_rows, in_row

    def anneal(rows0, rng, steps=50000):
        if time_up():
            return None
        rows = [row[:] for row in rows0]
        used_counts, diff_to_rows, in_row = build_state(rows)
        
        # Initial cost
        cost = sum(c * c for c in used_counts.values())
        T = 1.0
        T_min = 0.001
        best_local_cost = cost
        best_local_rows = [row[:] for row in rows]
        best_local_max = max(max(row) for row in rows)
        
        def refresh_best():
            nonlocal best_local_cost, best_local_rows, best_local_max
            if cost < best_local_cost or (cost == best_local_cost and max(max(row) for row in rows) < best_local_max):
                best_local_cost = cost
                best_local_max = max(max(row) for row in rows)
                best_local_rows = [row[:] for row in rows]
                
        refresh_best()
        
        for s in range(steps):
            if time_up():
                break
            
            r = rng.randrange(n_rows)
            row_vals = sorted(in_row[r])
            
            # Choose move
            if rng.random() < 0.5:
                # Swap: remove x1, add x2
                if len(row_vals) < 2:
                    continue
                i = rng.randrange(len(row_vals))
                x1 = row_vals[i]
                
                # Propose x2
                # x2 should be close to x1 or in a "free" spot
                off = rng.choice([-3, -2, -1, 1, 2, 3])
                x2 = x1 + off
                if x2 <= 0 or x2 > upper or x2 in in_row[r]:
                    continue
                
                row_vals_minus_x1 = [v for v in row_vals if v != x1]
                
                d_rm = delta_remove(r, x1, row_vals, used_counts)
                d_ad = delta_add(r, x2, row_vals_minus_x1, used_counts)
                delta_cost = d_rm + d_ad
                
                if delta_cost <= 0 or rng.random() < math.exp(-delta_cost / T):
                    # Apply move
                    # We need to apply carefully. 
                    # Remove x1
                    rows, used_counts, diff_to_rows, in_row = apply_move(r, x1, -1, rows, used_counts, diff_to_rows, in_row)
                    # Add x2
                    rows, used_counts, diff_to_rows, in_row = apply_move(r, x2, +1, rows, used_counts, diff_to_rows, in_row)
                    cost += delta_cost
                    refresh_best()
            else:
                # Shift: move x1 to x2 (x1 is removed, x2 is added, but x1 and x2 are distinct)
                # This is essentially the same as swap.
                # Let's try "Slide": pick x, move it by +/- 1, 2, 3.
                i = rng.randrange(len(row_vals))
                x1 = row_vals[i]
                off = rng.choice([-2, -1, 1, 2])
                x2 = x1 + off
                if x2 <= 0 or x2 > upper or x2 in in_row[r]:
                    continue
                
                row_vals_minus_x1 = [v for v in row_vals if v != x1]
                d_rm = delta_remove(r, x1, row_vals, used_counts)
                d_ad = delta_add(r, x2, row_vals_minus_x1, used_counts)
                delta_cost = d_rm + d_ad
                
                if delta_cost <= 0 or rng.random() < math.exp(-delta_cost / T):
                    rows, used_counts, diff_to_rows, in_row = apply_move(r, x1, -1, rows, used_counts, diff_to_rows, in_row)
                    rows, used_counts, diff_to_rows, in_row = apply_move(r, x2, +1, rows, used_counts, diff_to_rows, in_row)
                    cost += delta_cost
                    refresh_best()
            
            T = max(T_min, T * 0.9999)
            
            if cost == total_diffs:
                # Valid state found (all c(d)=1)
                m = max(max(row) for row in rows)
                if m < best_local_max:
                    best_local_max = m
                    best_local_cost = cost
                    best_local_rows = [row[:] for row in rows]
                    # Could stop early if we hit the lower bound?
                    # Or just continue to try to reduce max.
        
        return best_local_rows

    # 3. Run Annealing
    # Use multiple restarts
    rng = random.Random(42)
    starts = []
    for s in range(5):
        g = greedy_baseline(s)
        if g is not None:
            starts.append(g)
    if not starts:
        starts.append(best_rows)
    
    # Also add a perturbed start
    perturbed = [row[:] for row in best_rows]
    for r in range(n_rows):
        for i in range(len(perturbed[r])):
            if i > 0:
                perturbed[r][i] += rng.randint(-2, 2)
    # Sort and ensure validity (might break, but SA will fix or reject)
    # Actually, better to just use the greedy starts.
    
    attempts = 0
    for base in starts:
        if time_up():
            break
        for rep in range(2):
            if time_up():
                break
            res = anneal(base, random.Random(attempts * 101 + 7), steps=50000)
            if res is not None:
                if is_valid(res):
                    record(res)
            attempts += 1
            
    return best_rows
