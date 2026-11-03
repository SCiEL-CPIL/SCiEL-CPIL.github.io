import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Total number of differences needed
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
    # Helper: compute all positive differences in a row
    def row_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs
    
    # Helper: compute max entry
    def max_entry(rows):
        return max(max(r) for r in rows)
    
    # Helper: build used set from rows
    def build_used(rows):
        used = set()
        for r in rows:
            for d in row_diffs(r):
                used.add(d)
        return used
    
    # Greedy construction with randomization
    def greedy_construct(n_rows, k, used=None, randomize=False, p_smallest=0.75, window=4, max_x=200):
        if used is None:
            used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_diffs = set()
            for _ in range(k):
                x = row[-1] + 1
                forbidden = used | current_diffs
                valid_xs = []
                while x < max_x:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & forbidden):
                        valid_xs.append(x)
                        if len(valid_xs) >= window:
                            break
                    x += 1
                if not valid_xs:
                    return None
                if randomize:
                    if random.random() < p_smallest:
                        next_x = valid_xs[0]
                    else:
                        next_x = random.choice(valid_xs)
                else:
                    next_x = valid_xs[0]
                row.append(next_x)
                current_diffs |= {next_x - a for a in row[:-1]}
            rows.append(row)
            used |= current_diffs
        return rows
    
    # Compaction: try to shift elements down
    def compact_row(row, used, current_row_diffs, k):
        """Try to reduce the max entry of a row by shifting elements down."""
        n = len(row)
        # Work with a copy
        new_row = row[:]
        # Try to shift each element (from right to left) down
        for idx in range(n - 1, 0, -1):
            old_val = new_row[idx]
            # Try shifting down
            for shift in range(1, old_val):
                new_val = old_val - shift
                if new_val <= new_row[idx - 1]:
                    break
                # Check if new_val creates valid differences
                # Remove old diffs involving this element
                old_element_diffs = set()
                for j in range(n):
                    if j != idx:
                        d = abs(new_row[idx] - new_row[j])
                        old_element_diffs.add(d)
                
                # Check new differences
                new_element_diffs = set()
                valid = True
                for j in range(n):
                    if j != idx:
                        d = abs(new_val - new_row[j])
                        new_element_diffs.add(d)
                
                # All new diffs must not be in (used - old_element_diffs) and must be distinct
                forbidden = (used - old_element_diffs) | (current_row_diffs - old_element_diffs)
                if new_element_diffs & forbidden:
                    continue
                # Also check within new_element_diffs (they should be distinct, which they are since new_val != new_row[j])
                
                # Accept the shift
                new_row[idx] = new_val
                # Update current_row_diffs
                current_row_diffs -= old_element_diffs
                current_row_diffs |= new_element_diffs
                break
        return new_row
    
    # Initial greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_row_diffs = set()
        for _ in range(k):
            x = row[-1] + 1
            forbidden = used | current_row_diffs
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden):
                    row.append(x)
                    current_row_diffs |= new_diffs
                    break
                x += 1
        rows.append(row)
        used |= current_row_diffs
    
    best_rows = [r[:] for r in rows]
    best_max = max_entry(best_rows)
    current_max = best_max
    
    # Simulated Annealing
    T = 15.0
    cooling = 0.997
    min_T = 0.05
    
    while time.time() < deadline:
        # Pick a random row to regenerate
        r_idx = random.randint(0, n_rows - 1)
        
        # Remove diffs of row r_idx from used
        old_row = rows[r_idx]
        old_diffs = set(row_diffs(old_row))
        used -= old_diffs
        
        # Regenerate with randomization
        new_row = None
        # Try a few times to generate
        for attempt in range(3):
            new_row = greedy_construct(1, k, used, randomize=True, 
                                       p_smallest=0.7 + 0.1 * attempt, 
                                       window=3 + attempt, max_x=200)
            if new_row is not None:
                new_row = new_row[0]
                break
        
        if new_row is None:
            used |= old_diffs
            continue
        
        # Calculate new max
        new_max_val = max(new_row)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_max_val)
        
        delta = new_global_max - current_max
        
        if delta < 0:
            # Accept
            rows[r_idx] = new_row
            used |= set(row_diffs(new_row))
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            # Accept with probability
            if delta == 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)
            
            if random.random() < prob:
                rows[r_idx] = new_row
                used |= set(row_diffs(new_row))
                current_max = new_global_max
            else:
                # Rollback
                used |= old_diffs
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Final compaction pass
    if time.time() < deadline - 5.0:
        for r_idx in range(n_rows):
            row = rows[r_idx]
            current_row_diffs = set(row_diffs(row))
            other_diffs = set()
            for i in range(n_rows):
                if i != r_idx:
                    other_diffs |= set(row_diffs(rows[i]))
            
            new_row = compact_row(row, other_diffs, current_row_diffs, k)
            if max(new_row) < max(row):
                # Verify validity
                all_diffs = set()
                for i in range(n_rows):
                    if i != r_idx:
                        all_diffs |= set(row_diffs(rows[i]))
                new_diffs = set(row_diffs(new_row))
                if not (new_diffs & all_diffs) and len(new_diffs) == len(row_diffs(new_row)):
                    rows[r_idx] = new_row
                    if max_entry(rows) < best_max:
                        best_max = max_entry(rows)
                        best_rows = [r[:] for r in rows]
    
    return best_rows
