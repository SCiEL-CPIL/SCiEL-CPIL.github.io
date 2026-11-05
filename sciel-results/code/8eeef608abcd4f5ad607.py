"""Difference triangle set construction via randomized local search."""
import random
import time
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    time_limit = 110  # leave some margin
    
    def get_max_val(rows):
        return max(max(r) for r in rows)
    
    def compute_all_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def greedy_construct_minimal(rng):
        """Greedy construction picking smallest valid value each time."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        break
                    x += 1
                row.append(x)
                used |= new_diffs
            rows.append(row)
        return rows
    
    def rebuild_row(rows, row_idx, max_val, rng, other_diffs):
        """Rebuild a single row with max <= max_val, avoiding conflicts with other_diffs."""
        row = [0]
        local_diffs = set()
        for _ in range(k):
            min_x = row[-1] + 1
            if min_x > max_val:
                return None
            candidates = list(range(min_x, max_val + 1))
            rng.shuffle(candidates)
            found = False
            for x in candidates:
                new_diffs = {x - a for a in row}
                if len(new_diffs) != len(row):
                    continue
                if new_diffs & other_diffs or new_diffs & local_diffs:
                    continue
                row.append(x)
                local_diffs |= new_diffs
                found = True
                break
            if not found:
                return None
        return row
    
    # Get initial solution
    initial_rows = greedy_construct_minimal(random.Random(0))
    best_rows = [list(r) for r in initial_rows]
    best_max = get_max_val(best_rows)
    
    current_rows = [list(r) for r in initial_rows]
    current_max = best_max
    used = compute_all_diffs(current_rows)
    
    temperature = 20.0
    cooling_rate = 0.995
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit:
        if temperature < 0.5:
            break
        
        new_rows = [list(r) for r in current_rows]
        
        # Mutation: rebuild a random row with a tighter max bound
        row_idx = rng.randint(0, n_rows - 1)
        old_max = max(new_rows[row_idx])
        target_max = max(1, old_max - rng.randint(1, max(1, old_max // 4)))
        
        # Compute other rows' diffs
        other_diffs = set()
        for i, row in enumerate(new_rows):
            if i == row_idx:
                continue
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    other_diffs.add(row[b] - row[a])
        
        new_row = rebuild_row(new_rows, row_idx, target_max, rng, other_diffs)
        
        if new_row is not None:
            new_rows[row_idx] = new_row
            new_max = get_max_val(new_rows)
            delta = new_max - current_max
            
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                current_rows = new_rows
                current_max = new_max
                used = compute_all_diffs(current_rows)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in current_rows]
        
        # Occasionally try a full restart
        if rng.random() < 0.005:
            trial_rows = greedy_construct_minimal(rng)
            if trial_rows:
                trial_max = get_max_val(trial_rows)
                if trial_max < current_max:
                    current_rows = [list(r) for r in trial_rows]
                    current_max = trial_max
                    used = compute_all_diffs(current_rows)
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [list(r) for r in current_rows]
        
        temperature *= cooling_rate
    
    return best_rows
