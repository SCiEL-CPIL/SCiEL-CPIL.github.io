import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def get_row_mask(row):
        mask = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    def is_valid_row(row):
        # Check for internal conflicts
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def is_valid_all(rows):
        all_diffs = set()
        for r in rows:
            if not is_valid_row(r):
                return False
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # 1. Robust Greedy Initialization
    # We must ensure each row is a valid Golomb ruler (internal diffs distinct)
    # and global diffs are distinct.
    
    def generate_valid_row(existing_mask, max_val):
        row = [0]
        while len(row) < k + 1:
            found = False
            # Try random candidates in increasing order of value to keep it small
            # But we need to ensure internal validity.
            # A simple greedy: try x from row[-1]+1 to max_val
            # Check if x creates internal conflict with current row
            # AND if x creates global conflict with existing_mask
            
            # To speed up, we can pre-check internal conflicts
            # Internal conflict if x - row[i] == row[j] - row[i] for some i<j?
            # i.e., x - row[i] == d for some existing internal diff d involving row[i]?
            # Actually, just check if the set of new diffs (x - row[i]) intersects with existing internal diffs of row?
            # No, new diffs are x - row[i] for all i.
            # Existing internal diffs are row[j] - row[i].
            # Conflict if x - row[i] == row[j] - row[i'] for some i, i', j.
            
            # Simpler: Just build the row and check at the end? No, we need to build it incrementally.
            # Let's just try candidates and check full validity of the partial row + global.
            
            candidates = range(row[-1] + 1, max_val + 1)
            # Shuffle a bit to avoid getting stuck in specific patterns? 
            # For init, deterministic greedy is usually fine if we restart.
            
            for x in candidates:
                # Check internal validity of row + [x]
                new_row = row + [x]
                # Check if new_row has internal duplicate diffs
                # We can maintain a set of diffs for the current row to speed this up
                # But for k=7, rechecking is fast.
                
                # Optimization: Pre-compute diffs of current row
                # But we are in a tight loop. Let's just do the O(k^2) check.
                internal_ok = True
                diffs = set()
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        if d in diffs:
                            internal_ok = False
                            break
                        diffs.add(d)
                if not internal_ok:
                    continue
                
                # Check global conflict
                # New diffs are x - row[i] for i in range(len(row))
                # We need to check if any of these are in existing_mask
                # And also if any of these are in the *current row's* existing diffs? 
                # No, existing_mask is global. The current row's diffs are already in existing_mask?
                # Wait, in `generate_valid_row`, we pass `existing_mask` which contains diffs from *previous* rows.
                # The current row's diffs are NOT in `existing_mask` yet.
                # So we must check:
                # 1. New diffs (x - row[i]) do not conflict with existing_mask (other rows)
                # 2. New diffs (x - row[i]) do not conflict with current row's existing diffs (internal)
                
                # We already checked internal validity of the whole new_row, which covers (2).
                # Now check (1).
                
                global_conflict = False
                for i in range(len(row)):
                    d = x - row[i]
                    if (existing_mask >> d) & 1:
                        global_conflict = True
                        break
                
                if not global_conflict:
                    return new_row
            
            # If no candidate found, return None
            return None

    # Initial Generation
    best_rows = None
    best_max = float('inf')
    
    for attempt in range(20):
        if time.time() > deadline - 10: break
        
        rows = []
        global_mask = 0
        success = True
        
        for _ in range(n_rows):
            # Generate a row that doesn't conflict with global_mask
            # Max val estimate: 200 is safe for (4,7)
            new_row = generate_valid_row(global_mask, 200)
            if new_row is None:
                success = False
                break
            rows.append(new_row)
            # Update global mask with new row's diffs
            row_mask = get_row_mask(new_row)
            global_mask |= row_mask
        
        if success and is_valid_all(rows):
            curr_max = max(max(r) for r in rows)
            if curr_max < best_max:
                best_max = curr_max
                best_rows = [r[:] for r in rows]
        
        if best_rows is not None:
            # If we found a valid set, we can proceed to local search
            # But maybe try a few more initializations to get a better start?
            # For now, let's just take the first valid one or the best of few.
            pass

    if best_rows is None:
        # Fallback: Construct a valid set using a known method or very wide search
        # This should not happen for (4,7)
        # Create a simple valid set by using large gaps
        rows = []
        global_mask = 0
        for _ in range(n_rows):
            row = [0]
            base = 100 * (len(rows) + 1) # Large offset
            for i in range(1, k+1):
                row.append(base + i)
            rows.append(row)
            global_mask |= get_row_mask(row)
        best_rows = rows
        best_max = max(max(r) for r in best_rows)

    # Local Search: Simulated Annealing
    # State: rows
    # We will allow invalid states to escape local minima, but penalize them heavily.
    # Energy = (global_conflicts * 1000) + (internal_conflicts * 1000) + max_entry^2
    
    rows = [r[:] for r in best_rows]
    
    def calculate_energy(rows):
        # Returns (energy, max_entry, is_valid)
        all_diffs = set()
        internal_conflicts = 0
        global_conflicts = 0
        max_entry = 0
        
        for r in rows:
            row_diffs = set()
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in row_diffs:
                        internal_conflicts += 1
                    row_diffs.add(d)
                    if d in all_diffs:
                        global_conflicts += 1
                    all_diffs.add(d)
            max_entry = max(max_entry, max(r))
        
        # Energy function
        # We want to minimize max_entry, but we MUST have 0 conflicts.
        # If conflicts > 0, energy is huge.
        energy = max_entry * max_entry
        if internal_conflicts > 0:
            energy += internal_conflicts * 10000
        if global_conflicts > 0:
            energy += global_conflicts * 10000
            
        return energy, max_entry, (internal_conflicts == 0 and global_conflicts == 0)

    current_energy, current_max, current_valid = calculate_energy(rows)
    best_energy = current_energy
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    T = 500.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick a row and index to change
        r_idx = random.randint(0, n_rows - 1)
        j_idx = random.randint(1, k) # 0 is fixed
        
        old_v = rows[r_idx][j_idx]
        
        # Determine new value range
        # Bias towards smaller values to reduce max
        # But allow some exploration
        if random.random() < 0.8:
            # Try to shrink
            L = rows[r_idx][j_idx-1] + 1
            R = old_v
        else:
            # Try to expand or move
            L = rows[r_idx][j_idx-1] + 1
            R = old_v + 20
        
        if L > R:
            continue
            
        new_v = random.randint(L, R)
        
        if new_v == old_v:
            continue
            
        # Check strict increase
        if new_v <= rows[r_idx][j_idx-1]:
            continue
        if j_idx < k and new_v >= rows[r_idx][j_idx+1]:
            continue
            
        # Apply move
        rows[r_idx][j_idx] = new_v
        
        new_energy, new_max, new_valid = calculate_energy(rows)
        
        # Acceptance criteria
        if new_energy < current_energy:
            current_energy = new_energy
            rows[r_idx][j_idx] = new_v # Already applied
            if new_valid and new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in rows]
        else:
            delta = new_energy - current_energy
            # Metropolis criterion
            if random.random() < pow(2.718281828, -delta / T):
                current_energy = new_energy
                # Move stays
                if new_valid and new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in rows]
            else:
                # Reject move
                rows[r_idx][j_idx] = old_v
                
        T *= cooling
        if T < 1.0:
            T = 1.0

    return best_rows
