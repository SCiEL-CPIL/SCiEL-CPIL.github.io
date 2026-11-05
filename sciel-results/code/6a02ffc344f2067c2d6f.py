"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    def get_row_diffs(row):
        """Get set of all positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            row_diffs = get_row_diffs(row)
            if len(row_diffs) != k * (k + 1) // 2:
                return False
            if all_diffs & row_diffs:
                return False
            all_diffs |= row_diffs
        return True
    
    def greedy_construct():
        """Greedy construction: each row takes smallest next mark with unused differences."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Get initial solution via greedy
    rows = greedy_construct()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    # Try to improve with local search
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    current_rows = [r[:] for r in best_rows]
    current_max = best_max
    
    # Precompute current differences
    all_diffs = set()
    for row in current_rows:
        all_diffs |= get_row_diffs(row)
    
    temperature = best_max * 0.05
    cooling = 0.995
    min_temp = 0.001
    
    while time.time() - start_time < time_limit:
        if temperature < min_temp:
            break
        
        # Pick a random row and position to modify
        row_idx = random.randint(0, n_rows - 1)
        pos_idx = random.randint(1, k)  # Position 0 is fixed at 0
        
        old_val = current_rows[row_idx][pos_idx]
        
        # Determine valid range for new value
        lower = current_rows[row_idx][pos_idx - 1] + 1
        upper = current_rows[row_idx][pos_idx + 1] - 1 if pos_idx < k else old_val
        
        if upper < lower:
            temperature *= cooling
            continue
        
        # Try to reduce the value
        # Generate candidate values
        candidates = set()
        # Try the minimum possible
        candidates.add(lower)
        # Try values slightly below current
        for delta in range(1, min(10, old_val - lower + 1)):
            candidates.add(old_val - delta)
        # Try some random values in range
        for _ in range(10):
            candidates.add(random.randint(lower, upper))
        
        candidates = [c for c in candidates if lower <= c <= upper and c != old_val]
        if not candidates:
            temperature *= cooling
            continue
        
        new_val = random.choice(candidates)
        
        # Build the new row
        new_row = current_rows[row_idx][:]
        new_row[pos_idx] = new_val
        
        # Check strictly increasing
        if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
            temperature *= cooling
            continue
        
        # Compute differences for the new row
        new_row_diffs = get_row_diffs(new_row)
        if len(new_row_diffs) != k * (k + 1) // 2:
            temperature *= cooling
            continue
        
        # Get old differences for this row
        old_row_diffs = get_row_diffs(current_rows[row_idx])
        
        # Differences that are removed and added
        removed = old_row_diffs - new_row_diffs
        added = new_row_diffs - old_row_diffs
        
        # Check if any added difference conflicts with other rows or within new row
        # Other rows' differences
        other_diffs = all_diffs - old_row_diffs
        
        if added & other_diffs:
            temperature *= cooling
            continue
        
        # Check within new row: if any difference appears twice, it's already caught by len check
        # But we also need to ensure no duplicate within the new row itself
        # The len check above ensures all diffs are unique within the row
        
        # Compute new max
        new_max = 0
        for idx in range(n_rows):
            if idx == row_idx:
                new_max = max(new_max, new_row[-1])
            else:
                new_max = max(new_max, current_rows[idx][-1])
        
        # Acceptance criterion
        if new_max < current_max:
            # Accept and update
            all_diffs = (all_diffs - old_row_diffs) | new_row_diffs
            current_rows[row_idx] = new_row
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
        elif new_max == current_max:
            # Accept with some probability to explore
            if random.random() < 0.3:
                all_diffs = (all_diffs - old_row_diffs) | new_row_diffs
                current_rows[row_idx] = new_row
        else:
            # Reject or accept with small probability
            import math
            delta = new_max - current_max
            prob = math.exp(-delta / max(temperature, 0.01))
            if random.random() < prob:
                all_diffs = (all_diffs - old_row_diffs) | new_row_diffs
                current_rows[row_idx] = new_row
                current_max = new_max
        
        temperature *= cooling
    
    return best_rows

# EVOLVE-BLOCK-END
