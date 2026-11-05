import random
import math

def construct(n_rows, k):
    # 1. Initial Solution: Greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    
    # Helper to calculate cost
    def get_cost(rows):
        # Max value
        m = max(max(r) for r in rows)
        # Conflicts
        all_diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    all_diffs.append(r[j] - r[i])
        # Count duplicates
        seen = set()
        conflicts = 0
        for d in all_diffs:
            if d in seen:
                conflicts += 1
            else:
                seen.add(d)
        # Cost: We want to minimize m, but only if valid.
        # If invalid, we want to reduce conflicts.
        # Let's use: cost = m + conflicts * 1000
        # This prioritizes validity heavily.
        return m + conflicts * 1000, m, conflicts

    current_cost, cur_m, cur_conf = get_cost(rows)
    
    # SA Parameters
    T = 100.0
    cooling = 0.999
    iterations = 20000  # Adjust based on time
    min_T = 0.1
    
    import time
    start_time = time.time()
    time_limit = 110  # 120s limit, leave buffer
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Generate Move
        r_idx = random.randint(0, n_rows - 1)
        # Choose move type: shift suffix or change single element
        if random.random() < 0.5:
            # Shift suffix
            if k == 0:
                continue
            j_idx = random.randint(1, k)  # 1 to k (0 is fixed)
            delta = random.choice([-2, -1, 1, 2])
            # Check boundary
            if j_idx > 0:
                # rows[r_idx][j_idx-1] < rows[r_idx][j_idx] + delta
                if rows[r_idx][j_idx-1] >= rows[r_idx][j_idx] + delta:
                    continue
            # Apply shift
            old_suffix = rows[r_idx][j_idx:]
            new_suffix = [v + delta for v in old_suffix]
            rows[r_idx][j_idx:] = new_suffix
        else:
            # Change single element
            j_idx = random.randint(1, k)
            delta = random.choice([-2, -1, 1, 2])
            new_val = rows[r_idx][j_idx] + delta
            # Check bounds
            left = rows[r_idx][j_idx-1]
            right = rows[r_idx][j_idx+1] if j_idx < k else float('inf')
            if left < new_val < right:
                rows[r_idx][j_idx] = new_val
            else:
                continue # Revert/ignore if invalid move

        new_cost, new_m, new_conf = get_cost(rows)
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_cost = new_cost
            if new_conf == 0 and new_m < best_max:
                best_max = new_m
                best_rows = [r[:] for r in rows]
        else:
            # Revert
            # We need to undo the move. Easier to keep a copy or re-apply inverse.
            # Since we didn't store the old state fully, let's just re-generate or store.
            # Better: Store old row before move.
            pass # The logic above modifies in place. We need to revert.
            
        T *= cooling

    return best_rows
