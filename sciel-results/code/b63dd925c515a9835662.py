import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        """Check if rows are valid (strictly increasing, start at 0, unique diffs)."""
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_init(seed=None):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                # Try to find the next smallest valid number
                # We look in a window to allow some randomness
                found = False
                # Start from last+1, but allow skipping to avoid getting stuck in a tight cluster
                # Randomly decide how far to look
                max_search = 100
                
                # First pass: deterministic smallest
                for x in range(last + 1, last + max_search + 1):
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        found = True
                        break
                
                if not found:
                    # If stuck, expand search
                    x = last + max_search
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            row.append(x)
                            for a in row[:-1]:
                                used_diffs.add(x - a)
                            break
                
            rows.append(row)
            
        return rows

    def simulated_annealing(rows, time_limit):
        """
        Simulated Annealing to minimize the scope.
        Moves:
        1. Single element shift: Move an element to a new valid position.
        2. Row reconstruction: Rebuild a random row.
        """
        start = time.time()
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        # Maintain a set of all differences for fast checking
        # We need to update this set when rows change.
        # For efficiency in SA, we might calculate delta cost on the fly or maintain the set.
        # Maintaining the set is O(1) for checks but O(k^2) for updates.
        # Since k=7, k^2=49. This is fast enough.
        
        def get_all_diffs(rws):
            diffs = set()
            for r in rws:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        diffs.add(r[j] - r[i])
            return diffs

        current_diffs = get_all_diffs(current_rows)
        
        T = 50.0  # Initial temperature
        T_min = 0.1
        cooling_rate = 0.995
        
        while time.time() < start + time_limit and T > T_min:
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.7:
                # Move 1: Single element shift
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, row_len - 1)  # Don't move the 0
                
                old_val = current_rows[r_idx][c_idx]
                row = current_rows[r_idx]
                
                # Determine valid range for new value
                prev_val = row[c_idx - 1]
                next_val = row[c_idx + 1] if c_idx < row_len - 1 else old_val + 100
                
                # Try a few random candidates
                improved = False
                for _ in range(5):
                    # Bias towards smaller values to reduce scope
                    # Range: (prev_val, next_val)
                    # If c_idx is the last element, we can go up to current_scope + 10
                    if c_idx == row_len - 1:
                        high = min(best_scope + 10, old_val + 20)
                    else:
                        high = next_val - 1
                    
                    low = prev_val + 1
                    
                    if low > high:
                        continue
                        
                    new_val = random.randint(low, high)
                    if new_val == old_val:
                        continue
                    
                    # Check validity of new_val in this row
                    valid = True
                    # Check against other elements in the same row
                    for i in range(c_idx):
                        d = new_val - row[i]
                        if d in current_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    for j in range(c_idx + 1, len(row)):
                        d = row[j] - new_val
                        if d in current_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check internal uniqueness of new diffs
                    new_left = [new_val - row[i] for i in range(c_idx)]
                    new_right = [row[j] - new_val for j in range(c_idx + 1, len(row))]
                    if len(set(new_left + new_right)) != len(new_left + new_right):
                        continue
                    
                    # Calculate new scope
                    new_scope = current_scope
                    if new_val > current_scope:
                        new_scope = new_val
                    # If we are moving the current max, scope might decrease
                    # But we don't know if it's the unique max easily without checking all.
                    # Let's just compute it if it looks promising or always for simplicity in SA step?
                    # Computing scope is O(n*k). n=4, k=8. Very fast.
                    # Let's simulate the move to check scope properly.
                    
                    # Temporarily apply
                    row[c_idx] = new_val
                    
                    # Update diffs set? 
                    # For SA, we need to check if the move is valid globally.
                    # The checks above against `current_diffs` are sufficient if `current_diffs` 
                    # contained the diffs of the OLD state.
                    # However, we removed `old_val`. The diffs involving `old_val` are no longer present.
                    # So `current_diffs` still contains them.
                    # We need to be careful: `d in current_diffs` might be True because of the OLD element's diffs.
                    # This is a flaw in simple SA with global sets unless we update the set before checking.
                    
                    # Let's do a full validation for the candidate move to be safe, 
                    # or update the set.
                    # Given the small size, let's just do a full check of the new row against 
                    # the diffs of OTHER rows + new row's internal diffs.
                    
                    # Actually, the previous check `d in current_diffs` is dangerous.
                    # Let's revert and use a safer check.
                    row[c_idx] = old_val
                    
                    # Safe Check:
                    # 1. Remove old diffs involving old_val from a copy of the set? No, too slow.
                    # 2. Check new diffs against (current_diffs - old_diffs).
                    
                    old_diffs_set = set()
                    for i in range(c_idx):
                        old_diffs_set.add(old_val - row[i])
                    for j in range(c_idx + 1, len(row)):
                        old_diffs_set.add(row[j] - old_val)
                    
                    fixed_diffs = current_diffs - old_diffs_set
                    
                    valid = True
                    for i in range(c_idx):
                        d = new_val - row[i]
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    for j in range(c_idx + 1, len(row)):
                        d = row[j] - new_val
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    new_left = [new_val - row[i] for i in range(c_idx)]
                    new_right = [row[j] - new_val for j in range(c_idx + 1, len(row))]
                    if len(set(new_left + new_right)) != len(new_left + new_right):
                        continue
                    
                    # If valid, accept based on temperature
                    # Calculate new scope
                    temp_row = row[:]
                    temp_row[c_idx] = new_val
                    temp_scope = max(max(temp_row), max(max(r) for r in current_rows if r != row))
                    
                    delta = temp_scope - current_scope
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        # Accept
                        row[c_idx] = new_val
                        # Update current_diffs
                        current_diffs -= old_diffs_set
                        current_diffs.update(new_left + new_right)
                        current_scope = temp_scope
                        improved = True
                        break
            
            elif move_type < 0.9:
                # Move 2: Row Reconstruction
                r_idx = random.randint(0, n_rows - 1)
                old_row = current_rows[r_idx]
                
                # Remove old row diffs
                old_diffs_set = set()
                for i in range(len(old_row)):
                    for j in range(i+1, len(old_row)):
                        old_diffs_set.add(old_row[j] - old_row[i])
                
                fixed_diffs = current_diffs - old_diffs_set
                
                # Rebuild row
                new_row = [0]
                valid_row = True
                while len(new_row) < row_len:
                    last = new_row[-1]
                    found = False
                    # Try to find a value that doesn't conflict with fixed_diffs or new_row internal
                    # Bias towards small values
                    for x in range(last + 1, last + 50):
                        valid = True
                        for a in new_row:
                            d = x - a
                            if d in fixed_diffs or d in set(new_row): # d in set(new_row) is redundant if d>0 and a<x
                                valid = False
                                break
                        if valid:
                            new_row.append(x)
                            found = True
                            break
                    
                    if not found:
                        # Expand search
                        x = last + 50
                        while True:
                            x += 1
                            valid = True
                            for a in new_row:
                                d = x - a
                                if d in fixed_diffs:
                                    valid = False
                                    break
                            if valid:
                                new_row.append(x)
                                found = True
                                break
                    if not found:
                        valid_row = False
                        break
                
                if valid_row:
                    new_diffs_set = set()
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            new_diffs_set.add(new_row[j] - new_row[i])
                    
                    temp_scope = max(max(new_row), max(max(r) for r in current_rows if r != old_row))
                    delta = temp_scope - current_scope
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        # Accept
                        current_rows[r_idx] = new_row
                        current_diffs = (current_diffs - old_diffs_set) | new_diffs_set
                        current_scope = temp_scope
                        improved = True

            if improved:
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
            
            T *= cooling_rate

        return best_rows, best_scope

    def local_search_polish(rows, time_limit):
        """
        Hill climbing polish. Only accept improvements.
        """
        start = time.time()
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        
        def get_all_diffs(rws):
            diffs = set()
            for r in rws:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        diffs.add(r[j] - r[i])
            return diffs

        current_diffs = get_all_diffs(current_rows)
        
        while time.time() < start + time_limit:
            improved = False
            # Try to reduce each element
            for r_idx in range(n_rows):
                for c_idx in range(row_len - 1, 0, -1):  # From largest to smallest index
                    row = current_rows[r_idx]
                    val = row[c_idx]
                    prev_val = row[c_idx - 1]
                    
                    # Try to reduce val
                    # Find all diffs involving val
                    remove_set = set()
                    for i in range(c_idx):
                        remove_set.add(val - row[i])
                    for j in range(c_idx + 1, len(row)):
                        remove_set.add(row[j] - val)
                    
                    fixed_diffs = current_diffs - remove_set
                    
                    found = False
                    for new_val in range(val - 1, prev_val, -1):
                        valid = True
                        for i in range(c_idx):
                            d = new_val - row[i]
                            if d in fixed_diffs:
                                valid = False
                                break
                        if not valid:
                            continue
                        for j in range(c_idx + 1, len(row)):
                            d = row[j] - new_val
                            if d in fixed_diffs:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        new_left = [new_val - row[i] for i in range(c_idx)]
                        new_right = [row[j] - new_val for j in range(c_idx + 1, len(row))]
                        if len(set(new_left + new_right)) != len(new_left + new_right):
                            continue
                        
                        # Accept
                        row[c_idx] = new_val
                        current_diffs -= remove_set
                        current_diffs.update(new_left + new_right)
                        current_scope = get_scope(current_rows)
                        improved = True
                        found = True
                        break
                    
                    if found:
                        # Restart loop as indices might have shifted in value? No, indices fixed.
                        # But scope changed.
                        pass
            
            if not improved:
                break
        
        return current_rows, current_scope

    # Phase 1: Initialization
    best_rows = None
    best_scope = float('inf')
    
    # Run a few greedy inits
    for i in range(5):
        rows = greedy_init(seed=i)
        scope = get_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows

    # Phase 2: Simulated Annealing
    sa_time = 90.0
    if time.time() > deadline - 50:
        sa_time = deadline - time.time() - 10
    
    if sa_time > 10 and best_rows:
        sa_rows, sa_scope = simulated_annealing(best_rows, sa_time)
        if validate(sa_rows):
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows

    # Phase 3: Polish
    polish_time = min(40.0, max(0.0, deadline - time.time() - 5.0))
    if polish_time > 5.0 and best_rows:
        polished_rows, polished_scope = local_search_polish(best_rows, polish_time)
        if validate(polished_rows):
            if polished_scope < best_scope:
                best_scope = polished_scope
                best_rows = polished_rows

    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
