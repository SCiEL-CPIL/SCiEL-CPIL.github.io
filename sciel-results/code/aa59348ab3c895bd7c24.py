# EVOLVE-BLOCK-START
import sys
import time
import random

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # Improved Greedy Baseline
    def greedy_construct(n_rows, k, randomize=False, seed=0):
        used = set()
        rows = []
        rng = random.Random(seed) if randomize else None
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    if randomize and rng and rng.random() < 0.1:
                        # Small perturbation: try a slightly larger x next time?
                        # Actually, greedy is deterministic. 
                        # To randomize, we can skip a few valid candidates.
                        pass
            rows.append(row)
        return rows

    # Get a good upper bound using randomized greedy restarts
    best_greedy_rows = None
    best_greedy_max = float('inf')
    for i in range(10):
        # Standard greedy
        r = greedy_construct(n_rows, k)
        m = max(max(row) for row in r)
        if m < best_greedy_max:
            best_greedy_max = m
            best_greedy_rows = r
            
        # Randomized greedy: pick from first few valid candidates
        used = set()
        rows = []
        rng = random.Random(i)
        success = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                valid_candidates = []
                # Look ahead a bit
                for _look in range(5):
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        valid_candidates.append((x, new_diffs))
                        if len(valid_candidates) >= 5:
                            break
                if not valid_candidates:
                    success = False
                    break
                # Pick randomly from valid candidates
                x, nd = rng.choice(valid_candidates)
                row.append(x)
                used |= nd
            rows.append(row)
        
        if success:
            m = max(max(row) for row in rows)
            if m < best_greedy_max:
                best_greedy_max = m
                best_greedy_rows = rows

    if best_greedy_rows is None:
        best_greedy_rows = greedy_construct(n_rows, k)
        best_greedy_max = max(max(row) for row in best_greedy_rows)

    start_time = time.time()
    time_limit = 115.0
    
    def solve_csp(n_rows, k, max_entry, timeout, seed=0):
        deadline = time.time() + timeout
        used_bits = 0
        rows = []
        rng = random.Random(seed)
        
        # Precompute difference masks for quick conflict checking
        # For a candidate x with existing row entries, compute new diff bits
        def get_new_diff_bits(x, current_row, used_bits):
            new_bits = 0
            for a in current_row:
                d = x - a
                if d <= 0:
                    continue
                if (used_bits >> d) & 1:
                    return -1  # conflict
                new_bits |= (1 << d)
            return new_bits
        
        def build_row(current_row, used_bits, row_idx, deadline):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False
            
            # Collect valid candidates with their new difference bits
            candidates = []
            for x in range(min_x, max_x + 1):
                new_bits = get_new_diff_bits(x, current_row, used_bits)
                if new_bits >= 0:
                    # Score: prefer candidates that leave more room for future elements
                    # and don't use too many high differences
                    candidates.append((x, new_bits))
            
            if not candidates:
                return False
            
            # Heuristic ordering: try middle values first, then spread out
            # This helps with balance
            mid = (min_x + max_x) // 2
            candidates.sort(key=lambda c: (abs(c[0] - mid), rng.random()))
            
            # Try candidates in order
            for x, bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                if build_row(current_row, used_bits | bits, row_idx, deadline):
                    return True
                current_row.pop()
            
            return False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            current_row = [0]
            if build_row(current_row, used_bits, row_idx, deadline):
                rows.append(current_row)
                # Update used_bits from the completed row
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_bits |= (1 << d)
            else:
                return None
        
        return rows

    best_rows = best_greedy_rows
    best_max = best_greedy_max
    
    # Search for better M
    # We iterate M from lb to best_greedy_max
    # Allocate time based on M, with more aggressive search for promising M values
    for M in range(lb, best_greedy_max + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        
        if remaining_time < 0.5:
            break
        
        # Time budget allocation: exponential decay from lower bound
        # Smaller M gets more time since it's more constrained
        dist = M - lb
        # Use a smooth decay: budget decreases as M increases
        if dist == 0:
            budget = min(15.0, remaining_time * 0.4)
        elif dist <= 3:
            budget = min(8.0, remaining_time * 0.25)
        elif dist <= 8:
            budget = min(4.0, remaining_time * 0.15)
        else:
            budget = min(2.0, remaining_time * 0.08)
        
        if budget < 0.2:
            continue
            
        # Number of restarts: more for smaller M
        num_restarts = max(1, int(budget / 0.3))
        
        found = False
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            # Give each restart a portion of the budget
            restart_budget = max(0.1, budget / (num_restarts - seed))
            
            res = solve_csp(n_rows, k, M, restart_budget, seed=seed)
            if res is not None:
                best_rows = res
                best_max = M
                found = True
                break
        
        if found:
            # Found a solution at this M; try to improve with more restarts
            # but don't spend too much extra time
            extra_time = min(3.0, remaining_time * 0.1)
            for seed in range(3):
                if time.time() > start_time + time_limit - 2:
                    break
                restart_budget = extra_time / 3
                res = solve_csp(n_rows, k, M, restart_budget, seed=100 + seed)
                # Keep the best (smallest max entry)
                if res is not None:
                    m = max(max(row) for row in res)
                    if m < best_max:
                        best_rows = res
                        best_max = m
            break
            
    # Validate
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows
# EVOLVE-BLOCK-END
