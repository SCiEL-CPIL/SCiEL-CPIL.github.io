import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    def time_left():
        return deadline - time.time()

    # Helper to get all differences in a row
    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    # Greedy initialization
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            row_diffs = set()
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if x - a is in used or row_diffs for all a in row
                valid = True
                for a in row:
                    d = x - a
                    if d in used or d in row_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        row_diffs.add(x - a)
                    used.update(row_diffs)
            rows.append(row)
        return rows

    best_rows = greedy_init()
    best_scope = max(max(r) for r in best_rows)
    
    # State for search
    current_rows = [r[:] for r in best_rows]
    
    # Precompute diffs for current rows to speed up "used_other" calculation
    # We can just recompute it, k is small.
    
    T = 5.0 # Temperature for simulated annealing
    alpha = 0.995
    min_T = 0.1
    
    # Main Loop
    while time_left() > 2:
        # Pick a random row to rebuild
        ri = random.randint(0, n_rows - 1)
        old_row = current_rows[ri]
        old_max = max(old_row)
        
        # Calculate used diffs from all OTHER rows
        used_other = set()
        for j in range(n_rows):
            if j != ri:
                used_other.update(get_diffs(current_rows[j]))
        
        # Attempt to rebuild row ri
        # We want to find a row of length k+1, starting at 0, strictly increasing
        # such that all internal diffs are unique and not in used_other
        # We use randomized greedy to explore different shapes
        
        success = False
        new_row = None
        new_max = 0
        
        # Try a few times with different random seeds/jitters
        # Since k is small, we can try a few different random walks
        for attempt in range(20):
            row = [0]
            row_diffs = set()
            x = 0
            valid_row = True
            
            # Jitter increases with attempt to explore more diverse rows
            # But we want to keep the max small, so we bias towards small x
            # A simple way: with prob p, skip the smallest valid x
            p_skip = 0.1 + 0.05 * attempt
            
            while len(row) < k + 1:
                found = False
                # Look ahead to find valid candidates
                # We start from x+1
                start_x = x + 1
                # Limit search to a reasonable bound to avoid infinite loops
                # If we can't find one within a range, this attempt fails
                # A good bound is current best scope + some margin
                # But for local search, we just want *a* valid row
                # Let's scan up to 2000
                for trial_x in range(start_x, start_x + 2000):
                    # Check validity
                    is_valid = True
                    for a in row:
                        d = trial_x - a
                        if d in used_other or d in row_diffs:
                            is_valid = False
                            break
                    
                    if is_valid:
                        # Decide whether to take this or skip
                        # If we skip, we need to be careful not to skip too many
                        # We only skip if we haven't found the *first* valid one yet?
                        # No, greedy takes the first. Randomized greedy might skip the first.
                        # If we skip, we continue to the next trial_x.
                        
                        # To maintain "increasing" property, we just append if we accept.
                        # If we skip, we don't append, and continue loop.
                        
                        if random.random() > p_skip:
                            # Accept
                            row.append(trial_x)
                            for a in row[:-1]:
                                row_diffs.add(trial_x - a)
                            x = trial_x
                            found = True
                            break
                        # Else skip and try next trial_x
                    else:
                        # Not valid, just continue
                        pass
                
                if not found:
                    valid_row = False
                    break
            
            if valid_row:
                new_row = row
                new_max = max(new_row)
                success = True
                # We can break early if we find a row with max < old_max
                if new_max < old_max:
                    break
        
        if success:
            # Evaluate move
            # New global scope if we accept
            # Other rows' maxes are fixed.
            other_max = max(max(current_rows[j]) for j in range(n_rows) if j != ri) if n_rows > 1 else 0
            new_global_scope = max(other_max, new_max)
            old_global_scope = max(other_max, old_max)
            
            # Acceptance criteria
            if new_max < old_max:
                # Improvement in row max, likely improvement in global scope
                accept = True
            elif new_max == old_max:
                # Neutral for this row, neutral for global scope
                # Accept with high probability to explore
                accept = random.random() < 0.5
            else:
                # new_max > old_max
                # Global scope might increase
                # Accept with probability exp(-delta / T)
                delta = new_max - old_max
                if delta < 50: # Don't accept huge jumps
                    prob = math.exp(-delta / T)
                    accept = random.random() < prob
                else:
                    accept = False
            
            if accept:
                current_rows[ri] = new_row
                cur_scope = max(max(r) for r in current_rows)
                if cur_scope < best_scope:
                    best_scope = cur_scope
                    best_rows = [r[:] for r in current_rows]
        
        # Cool down
        T *= alpha
        if T < min_T:
            T = min_T

    return best_rows
