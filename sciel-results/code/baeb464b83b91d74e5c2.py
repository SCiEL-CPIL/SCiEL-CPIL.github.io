import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0

    total_diffs = n_rows * (k * (k + 1) // 2)
    lb = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2.0))

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

    def solve(upper, seed):
        rng = random.Random(seed)
        rows = [[] for _ in range(n_rows)]
        used_diffs = set()
        
        # We will fill row by row.
        # To add randomness, we can shuffle the order of rows? No, rows are symmetric.
        # We can randomize the candidate selection order.
        
        def rec(row_idx, pos_idx):
            if time_up():
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos_idx == 0:
                rows[row_idx].append(0)
                return rec(row_idx, 1)
            
            if pos_idx == k + 1:
                return rec(row_idx + 1, 0)
            
            last_val = rows[row_idx][pos_idx - 1]
            
            # Generate candidates
            # We want to try small values first to keep max low, but randomize to avoid local traps.
            # Let's create a list of candidates from last_val + 1 to upper.
            # If upper is large, this is expensive.
            # Optimization: Only try candidates up to a certain limit?
            # Or, try a random subset?
            
            # Let's try iterating, but if we fail many times, we might be stuck.
            # For now, simple iteration is fine for small n.
            
            for x in range(last_val + 1, upper + 1):
                valid = True
                # Check diffs with previous elements in this row
                for i in range(pos_idx):
                    d = x - rows[row_idx][i]
                    if d in used_diffs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Place x
                rows[row_idx].append(x)
                new_diffs = []
                for i in range(pos_idx):
                    d = x - rows[row_idx][i]
                    used_diffs.add(d)
                    new_diffs.append(d)
                
                if rec(row_idx, pos_idx + 1):
                    return True
                
                # Backtrack
                rows[row_idx].pop()
                for d in new_diffs:
                    used_diffs.remove(d)
                    
            return False

        if rec(0, 0):
            return rows
        return None

    best_rows = None
    best_max = float('inf')

    def record(rows):
        nonlocal best_rows, best_max
        if rows is None:
            return
        if not is_valid(rows):
            return
        m = max(max(row) for row in rows)
        if m < best_max:
            best_max = m
            best_rows = [row[:] for row in rows]

    # Try increasing upper bounds
    # Start from lb
    # lb is ~112.
    # We try lb, lb+1, ..., up to a limit.
    # But backtracking might be slow.
    # Let's try a few specific upper bounds with multiple seeds.
    
    upper_bounds = [lb + i for i in range(0, 50)] # lb to lb+49
    
    # Shuffle the upper bounds? No, we want the smallest.
    # But we should try multiple seeds for each bound.
    
    for upper in upper_bounds:
        if time_up():
            break
        for seed in range(10):
            if time_up():
                break
            res = solve(upper, seed)
            if res is not None:
                record(res)
                # If we found a solution, we can stop?
                # Since we are iterating upper from low to high, the first found is the best for this upper.
                # And since we go low to high, it's the global best found so far.
                # We can break out of both loops.
                return best_rows
    
    # If no solution found in the tight range, try a looser range
    for upper in [lb + 50, lb + 100, lb + 200, lb + 500]:
        if time_up():
            break
        for seed in range(20):
            if time_up():
                break
            res = solve(upper, seed)
            if res is not None:
                record(res)
                return best_rows

    # Fallback: Greedy with very large upper
    if best_rows is None:
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                for x in range(row[-1] + 1, 2000):
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                        found = True
                        break
                if not found:
                    break
            if len(row) < k + 1:
                # Emergency fallback: return a dummy valid structure if possible
                # For n=4, k=7, we can hardcode a known valid one if we have one.
                # But let's assume greedy works with 2000.
                # If it fails, we return a list of lists of 0s? No, must be increasing.
                # Let's just return None? No, must return list.
                # Let's try to construct a trivial one:
                # Row 0: 0, 1, 2, 3, 4, 5, 6, 7
                # Row 1: 0, 100, 200, 300, 400, 500, 600, 700
                # Row 2: 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000
                # Row 3: 0, 10000, 20000, 30000, 40000, 50000, 60000, 70000
                # Diffs are distinct?
                # Row 0 diffs: 1..7
                # Row 1 diffs: 100..700, 100..600 etc. All multiples of 100.
                # Row 0 diffs are 1-7. Row 1 diffs are 100-700. No overlap.
                # Row 2 diffs are 1000-7000. No overlap with 1-700.
                # Row 3 diffs are 10000-70000. No overlap.
                # So this is valid.
                rows = [
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [0, 100, 200, 300, 400, 500, 600, 700],
                    [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000],
                    [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000]
                ]
                # Adjust for n_rows, k
                # This hardcode is only for n=4, k=7.
                # For generality, we should scale.
                base = 1
                for i in range(n_rows):
                    row = [0]
                    for j in range(1, k+1):
                        row.append(j * base)
                    rows.append(row)
                    base *= 100
                break
            rows.append(row)
        record(rows)

    return best_rows

