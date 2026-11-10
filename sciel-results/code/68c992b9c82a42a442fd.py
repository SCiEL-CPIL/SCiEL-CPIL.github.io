import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110
    
    # 1. Generate initial valid solution
    # Use a greedy approach but optimized
    def generate_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            # Try to keep numbers small
            # We can use a window of candidates
            while len(row) < k + 1:
                found = False
                # Try candidates from last+1 upwards
                # Limit search to reasonable bound, e.g., 200
                for x in range(last + 1, last + 100):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        last = x
                        found = True
                        break
                if not found:
                    return None # Failed
            rows.append(row)
        return rows

    initial_rows = generate_greedy()
    if initial_rows is None:
        # Fallback: simpler greedy with larger window
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

    best_rows = [r[:] for r in initial_rows]
    best_max = max(max(r) for r in best_rows)
    
    # 2. Simulated Annealing
    current_rows = [r[:] for r in initial_rows]
    current_max = best_max
    
    # Precompute differences for each row to speed up checks?
    # Or just recompute on the fly?
    # For k=5 (6 elements), diffs are 15 per row.
    # Total 105 diffs.
    # Checking collision:
    # When changing x_{r,j}, we remove old diffs involving x_{r,j} and add new ones.
    
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    # We maintain a global set of all differences?
    # If we change one row, we need to remove its old diffs and add new ones.
    # And check against diffs of OTHER rows.
    
    all_diffs = set()
    row_diffs = []
    for r in current_rows:
        d = get_diffs(r)
        row_diffs.append(d)
        all_diffs |= d
        
    T = 100.0
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        if T < 0.1:
            break
            
        r = random.randint(0, n_rows - 1)
        # Pick an index to change. 
        # Changing the last element is most impactful for max.
        # But changing earlier elements allows more flexibility.
        # Let's pick random j from 1 to k (index 0 is fixed at 0)
        j = random.randint(1, k)
        
        old_val = current_rows[r][j]
        
        # Determine valid range for new_val
        # Must be > current_rows[r][j-1]
        # Must be < current_rows[r][j+1] if j < k
        low = current_rows[r][j-1] + 1
        high = current_rows[r][j+1] - 1 if j < k else old_val + 10 # Allow increasing?
        
        # We want to minimize max, so we should try values <= current_max
        # But sometimes we need to increase to allow others to decrease?
        # SA allows worsening moves.
        
        # Let's try a random value in [low, high]
        # To be efficient, try a few candidates
        found_valid = False
        new_val = None
        
        # Try to find a valid new_val
        # We need to check if new diffs collide with (all_diffs - row_diffs[r])
        other_diffs = all_diffs - row_diffs[r]
        
        # Generate candidates
        # If we want to reduce max, try values near low.
        # If we want to explore, try random.
        
        # Let's try a window around old_val and low
        candidates = []
        # Try values that might reduce the max
        if old_val > current_max:
            # Try to bring it down
            candidates = list(range(low, min(high, old_val) + 1, 5))
        else:
            # Try random moves
            candidates = [random.randint(low, high) for _ in range(10)]
            # Also try some local moves
            candidates.extend([old_val - 1, old_val + 1, low, high])
            
        random.shuffle(candidates)
        
        for val in candidates:
            if val < low or val > high:
                continue
            
            # Construct new row
            new_row = current_rows[r][:]
            new_row[j] = val
            
            # Check order
            if j > 0 and new_row[j] <= new_row[j-1]:
                continue
            if j < k and new_row[j] >= new_row[j+1]:
                continue
            
            # Calculate new diffs for this row
            new_diffs = get_diffs(new_row)
            
            # Check collision with other rows
            if new_diffs & other_diffs:
                continue
            
            # Check collision within new_row (already ensured by get_diffs set size? 
            # No, get_diffs returns a set, so duplicates within row are collapsed.
            # But we need distinct differences OVER ALL ROWS.
            # Does the problem require distinct differences WITHIN a row?
            # "all differences between two entries of the same row are distinct over all rows"
            # This phrasing is ambiguous.
            # Standard "Difference Set" requires all pairwise differences to be distinct.
            # If it allows duplicates within a row but not across rows, it's different.
            # "distinct over all rows" usually implies the global set is distinct.
            # If a row has duplicate differences, say 1 and 1, the global set has one 1.
            # But usually "Golomb Ruler" requires distinct differences within the ruler.
            # Let's assume the standard: ALL differences (global) are distinct.
            # This implies differences within a row must also be distinct.
            
            # So, we need len(new_diffs) == len(new_row)*(len(new_row)-1)//2
            # And new_diffs & other_diffs == empty set.
            
            if len(new_diffs) != len(new_row)*(len(new_row)-1)//2:
                continue
            
            # Valid move
            found_valid = True
            new_val = val
            break
        
        if not found_valid:
            T *= cooling
            continue
            
        # Calculate new max
        new_max = max(max(row) for row in current_rows if row is not current_rows[r])
        if new_val > new_max:
            new_max = new_val
            
        delta = new_max - current_max
        
        # Acceptance probability
        if delta < 0:
            accept = True
        else:
            if T > 0:
                p = math.exp(-delta / T)
                accept = random.random() < p
            else:
                accept = False
        
        if accept:
            # Update state
            # Remove old diffs
            all_diffs -= row_diffs[r]
            # Add new diffs
            all_diffs |= new_diffs
            row_diffs[r] = new_diffs
            current_rows[r] = new_row
            current_max = new_max
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [row[:] for row in current_rows]
                
        T *= cooling

    return best_rows
