# EVOLVE-BLOCK-START
import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses Simulated Annealing with multi-start greedy initialization and
    a final high-precision polish phase.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 120s limit

    def time_up():
        return time.time() > deadline

    total_diffs_needed = n_rows * k * (k + 1) // 2

    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    # 1. Multi-start Greedy with randomized candidate selection
    def greedy_construct(seed, max_limit=None):
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            
            for pos in range(k):
                last = row[-1]
                remaining = k - pos
                if max_limit is not None:
                    max_x = max_limit - (remaining - 1)
                else:
                    max_x = 20000
                    
                if last + 1 > max_x:
                    return None
                
                # Collect valid candidates from a window
                valid_candidates = []
                # Search a window of size proportional to remaining elements
                window_size = min(50, max_x - last)
                for x in range(last + 1, min(last + 1 + window_size, max_x + 1)):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & local_used) and not (new_diffs & used):
                        valid_candidates.append((x, new_diffs))
                        if len(valid_candidates) >= 15:
                            break
                
                if not valid_candidates:
                    # Try wider search if local window failed
                    for x in range(last + 1, max_x + 1):
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & local_used) and not (new_diffs & used):
                            valid_candidates.append((x, new_diffs))
                            if len(valid_candidates) >= 10:
                                break
                
                if not valid_candidates:
                    return None
                
                # Random selection from candidates for diversity
                idx = rng.randint(0, len(valid_candidates) - 1)
                x, new_diffs = valid_candidates[idx]
                row.append(x)
                local_used.update(new_diffs)
            
            rows.append(row)
            used.update(local_used)
        
        return rows, used

    # Initial best solution via multi-start
    best_rows = None
    best_max = float('inf')
    current_used = None
    
    for seed in range(80):
        if time_up():
            break
        res = greedy_construct(seed)
        if res:
            rows, used = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                current_used = used.copy()

    if best_rows is None:
        # Deterministic fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            for pos in range(k):
                last = row[-1]
                for x in range(last + 1, 50000):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & local_used) and not (new_diffs & used):
                        row.append(x)
                        local_used.update(new_diffs)
                        break
            rows.append(row)
            used.update(local_used)
        best_rows = rows
        best_max = get_max(best_rows)
        current_used = used.copy()

    # 2. Simulated Annealing with rebuild-based moves
    current_rows = [r[:] for r in best_rows]
    current_max = best_max

    rng = random.Random(123)
    
    def generate_random_row(forbidden_diffs, rng, max_val_limit, bias=0.5):
        """Generate a valid row avoiding forbidden_diffs, with max <= max_val_limit."""
        row = [0]
        local_used = set()
        
        for pos in range(k):
            last = row[-1]
            remaining = k - pos
            max_x_allowed = max_val_limit - (remaining - 1)
            
            if last + 1 > max_x_allowed:
                return None
            
            valid_candidates = []
            # Search window
            limit = min(max_x_allowed, last + 80)
            for x in range(last + 1, limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= 20:
                        break
            
            if not valid_candidates:
                # Wider search
                for x in range(last + 1, max_x_allowed + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                        valid_candidates.append((x, new_diffs))
                        if len(valid_candidates) >= 15:
                            break
            
            if not valid_candidates:
                return None
            
            # Selection with bias
            if rng.random() < bias:
                idx = rng.randint(0, len(valid_candidates) - 1)
            else:
                idx = 0  # Greedy: pick smallest
            x, new_diffs = valid_candidates[idx]
            row.append(x)
            local_used.update(new_diffs)
        
        return row

    # SA parameters
    T_start = max(5.0, current_max * 0.4)
    T_end = 0.01
    cooling_rate = 0.99985
    
    T = T_start
    no_improve = 0
    max_no_improve = 15000
    
    while not time_up() and no_improve < max_no_improve:
        # Pick row to replace
        i = rng.randint(0, n_rows - 1)
        old_row = current_rows[i]
        old_diffs = get_diffs(old_row)
        
        current_used.difference_update(old_diffs)
        
        # Determine target
        if max(old_row) == current_max:
            # This row holds the global max; try to reduce
            target_max = current_max - 1
            if rng.random() < 0.15:
                target_max = current_max  # Reshuffle at same level
        else:
            # Not the bottleneck; reshuffle or try to reduce
            if rng.random() < 0.3:
                target_max = max(k + 1, max(old_row) - 2)
            else:
                target_max = current_max
        
        new_row = generate_random_row(current_used, rng, target_max, bias=0.6)
        
        if new_row is None and target_max < current_max:
            new_row = generate_random_row(current_used, rng, current_max, bias=0.5)
        
        if new_row is None:
            current_used.update(old_diffs)
            no_improve += 1
            T *= cooling_rate
            continue
        
        new_diffs = get_diffs(new_row)
        
        if new_diffs & current_used:
            current_used.update(old_diffs)
            no_improve += 1
            T *= cooling_rate
            continue
        
        # Evaluate
        current_rows[i] = new_row
        new_global_max = get_max(current_rows)
        delta = new_global_max - current_max
        
        if delta <= 0:
            current_used.update(new_diffs)
            current_max = new_global_max
            no_improve = 0
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
        else:
            prob = math.exp(-delta / T) if T > 1e-9 else 0.0
            if rng.random() < prob:
                current_used.update(new_diffs)
                current_max = new_global_max
                no_improve += 1
            else:
                current_rows[i] = old_row
                current_used.update(old_diffs)
                no_improve += 1
        
        T *= cooling_rate

    # 3. Final high-precision polish: systematically try to reduce each row
    # Focus on rows that hold the global max
    polish_start = time.time()
    polish_deadline = start_time + 114.0
    
    current_rows = [r[:] for r in best_rows]
    current_used = set()
    for r in current_rows:
        current_used.update(get_diffs(r))
    current_max = get_max(current_rows)
    
    polish_rng = random.Random(999)
    
    polish_no_improve = 0
    max_polish_no_improve = 5000
    
    while time.time() < polish_deadline and polish_no_improve < max_polish_no_improve:
        # Find rows with max == current_max
        max_rows = [i for i in range(n_rows) if max(current_rows[i]) == current_max]
        
        if not max_rows:
            break
        
        # Try to reduce each max-row
        improved = False
        for i in max_rows:
            if time.time() > polish_deadline:
                break
            
            old_row = current_rows[i]
            old_diffs = get_diffs(old_row)
            current_used.difference_update(old_diffs)
            
            # Try to rebuild with max = current_max - 1
            target = current_max - 1
            if target >= k + 1:
                new_row = generate_random_row(current_used, polish_rng, target, bias=0.3)
                if new_row is None:
                    # Try with slightly higher bias
                    new_row = generate_random_row(current_used, polish_rng, target, bias=0.7)
                
                if new_row is not None:
                    new_diffs = get_diffs(new_row)
                    if not (new_diffs & current_used):
                        current_rows[i] = new_row
                        current_used.update(new_diffs)
                        current_max = get_max(current_rows)
                        improved = True
                        polish_no_improve = 0
                        if current_max < best_max:
                            best_max = current_max
                            best_rows = [r[:] for r in current_rows]
                        break  # Re-evaluate max_rows
                    else:
                        current_used.update(old_diffs)
                else:
                    current_used.update(old_diffs)
            else:
                current_used.update(old_diffs)
        
        if not improved:
            polish_no_improve += 1
    
    # Final validation
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
        # Fallback to greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            for pos in range(k):
                last = row[-1]
                for x in range(last + 1, 50000):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & local_used) and not (new_diffs & used):
                        row.append(x)
                        local_used.update(new_diffs)
                        break
            rows.append(row)
            used.update(local_used)
        best_rows = rows

    return best_rows

# EVOLVE-BLOCK-END
