import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    num_entries = k + 1
    
    def time_left():
        return deadline - time.time()
    
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_entries:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    best_rows = greedy()
    best_scope = get_scope(best_rows)
    
    # Simulated annealing with incremental diff tracking
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    
    # Precompute all diffs
    all_diffs = set()
    for row in current_rows:
        for i in range(num_entries):
            for j in range(i+1, num_entries):
                all_diffs.add(row[j] - row[i])
    
    T = 5.0
    T_min = 0.005
    cooling = 0.9995
    
    while time_left() > 5.0 and T > T_min:
        # Generate move
        row_idx = random.randint(0, n_rows - 1)
        entry_idx = random.randint(1, num_entries - 1)
        
        old_val = current_rows[row_idx][entry_idx]
        
        min_allowed = current_rows[row_idx][entry_idx - 1] + 1
        max_allowed = current_rows[row_idx][entry_idx + 1] - 1 if entry_idx + 1 < num_entries else best_scope
        
        if max_allowed < min_allowed:
            T *= cooling
            continue
        
        # Try to reduce: bias toward smaller values
        if random.random() < 0.7:
            new_val = random.randint(min_allowed, max(min_allowed + 5, max_allowed))
        else:
            new_val = random.randint(min_allowed, max_allowed)
        
        if new_val == old_val:
            T *= cooling
            continue
        
        # Compute diffs affected by this entry
        affected_diffs = set()
        for i in range(num_entries):
            if i != entry_idx:
                if i < entry_idx:
                    affected_diffs.add(old_val - current_rows[row_idx][i])
                else:
                    affected_diffs.add(current_rows[row_idx][i] - old_val)
        
        # New diffs
        new_diffs = set()
        for i in range(num_entries):
            if i != entry_idx:
                if i < entry_idx:
                    new_diffs.add(new_val - current_rows[row_idx][i])
                else:
                    new_diffs.add(current_rows[row_idx][i] - new_val)
        
        # Check validity: new_diffs should not be in (all_diffs - affected_diffs)
        remaining = all_diffs - affected_diffs
        if new_diffs & remaining:
            T *= cooling
            continue
        
        # Apply
        current_rows[row_idx][entry_idx] = new_val
        all_diffs = (all_diffs - affected_diffs) | new_diffs
        new_scope = get_scope(current_rows)
        
        delta = new_scope - current_scope
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            # Revert
            current_rows[row_idx][entry_idx] = old_val
            all_diffs = (all_diffs - new_diffs) | affected_diffs
        
        T *= cooling
    
    return best_rows
