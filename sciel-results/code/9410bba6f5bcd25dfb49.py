import time
import random
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110
    
    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                diffs = {x - a for a in row}
                if not (diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(diffs)
            rows.append(row)
        return rows

    def init_state(rows):
        counts = {}
        conflict_diffs = set()
        row_maxes = [max(r) for r in rows]
        penalty = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in counts:
                        counts[d] += 1
                        if counts[d] == 2:
                            conflict_diffs.add(d)
                        penalty += 1 # count went from c to c+1, penalty increases by 1 if c>=1
                    else:
                        counts[d] = 1
        # Wait, penalty logic:
        # If count is 1, penalty contribution is 0.
        # If count is 2, penalty contribution is 1.
        # If count is 3, penalty contribution is 2.
        # So when adding a diff:
        # If old count was 0 -> new 1: penalty +0
        # If old count was 1 -> new 2: penalty +1
        # If old count was 2 -> new 3: penalty +1
        # So penalty increases by 1 if old count >= 1.
        
        # Let's re-calculate penalty properly in init
        penalty = 0
        for c in counts.values():
            if c > 1:
                penalty += c - 1
                
        M = max(row_maxes)
        return counts, conflict_diffs, row_maxes, penalty, M

    def get_M_from_maxes(row_maxes):
        return max(row_maxes)

    best_rows = None
    best_M = float('inf')
    
    num_restarts = 80
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        curr_rows = greedy_init(seed=random.randint(0, 1000000))
        curr_counts, curr_conflicts, curr_row_maxes, curr_penalty, curr_M = init_state(curr_rows)
        
        if curr_penalty == 0 and curr_M < best_M:
            best_M = curr_M
            best_rows = copy.deepcopy(curr_rows)
            
        max_iter = 50000
        for _ in range(max_iter):
            if time.time() > deadline - 2:
                break
                
            if curr_penalty == 0:
                # Try to reduce scope
                r_idx = random.randint(0, n_rows - 1)
                row = curr_rows[r_idx]
                # Pick a random index to decrease
                # Bias towards larger indices
                j_idx = random.randint(1, k)
                old_val = row[j_idx]
                low = row[j_idx - 1] + 1
                if old_val <= low:
                    continue
                
                # Try to decrease by a small amount
                # Random step
                max_step = old_val - low
                step = random.randint(1, min(5, max_step))
                new_val = old_val - step
                
                # Check if new_val creates conflicts
                # We need to simulate
                # Remove old diffs
                removed_diffs = []
                for i in range(j_idx):
                    d = old_val - row[i]
                    removed_diffs.append(d)
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - old_val
                    removed_diffs.append(d)
                    
                # Add new diffs
                added_diffs = []
                for i in range(j_idx):
                    d = new_val - row[i]
                    added_diffs.append(d)
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - new_val
                    added_diffs.append(d)
                    
                # Calculate delta penalty
                delta_pen = 0
                # Remove
                for d in removed_diffs:
                    c = curr_counts[d]
                    # c -> c-1
                    # penalty change: max(0, c-1-1) - max(0, c-1)
                    # if c==1: 0 - 0 = 0
                    # if c==2: 1 - 1 = 0 (wait, c-1=1, max(0,0)=0. c=2, max(0,1)=1. 0-1=-1)
                    # if c>2: (c-2) - (c-1) = -1
                    if c > 1:
                        delta_pen -= 1
                        
                    # Update conflict set
                    if c == 2:
                        curr_conflicts.discard(d)
                    # else if c > 2, still conflict
                    
                # Add
                for d in added_diffs:
                    c = curr_counts.get(d, 0)
                    # c -> c+1
                    # penalty change: max(0, c+1-1) - max(0, c-1)
                    # if c==0: 0 - 0 = 0
                    # if c==1: 1 - 0 = 1
                    # if c>1: (c) - (c-1) = 1
                    if c >= 1:
                        delta_pen += 1
                        
                    # Update conflict set
                    if c == 1:
                        curr_conflicts.add(d)
                    # else if c==0, no conflict yet
                    
                # Apply changes if beneficial (delta_pen <= 0)
                if delta_pen <= 0:
                    # Commit
                    for d in removed_diffs:
                        curr_counts[d] -= 1
                        if curr_counts[d] == 0:
                            del curr_counts[d]
                            
                    for d in added_diffs:
                        if d in curr_counts:
                            curr_counts[d] += 1
                        else:
                            curr_counts[d] = 1
                            
                    row[j_idx] = new_val
                    curr_penalty += delta_pen
                    
                    # Update row max
                    if new_val > curr_row_maxes[r_idx]:
                        curr_row_maxes[r_idx] = new_val
                    else:
                        # If we decreased the max, we need to find new max
                        if old_val == curr_row_maxes[r_idx]:
                            curr_row_maxes[r_idx] = max(row)
                            
                    curr_M = get_M_from_maxes(curr_row_maxes)
                    
                    if curr_M < best_M:
                        best_M = curr_M
                        best_rows = copy.deepcopy(curr_rows)
                        
            else:
                # Resolve conflict
                if not curr_conflicts:
                    # Should not happen if penalty > 0, but safety check
                    continue
                    
                d = random.choice(list(curr_conflicts))
                # Find a pair with this difference
                # We need to search for a pair. This is O(N*K^2)
                # To speed up, we could maintain a map diff -> list of pairs, but that's complex to update.
                # For now, linear scan is okay if K is small.
                
                found = False
                for r_idx in range(n_rows):
                    row = curr_rows[r_idx]
                    for i in range(k):
                        for j in range(i+1, k+1):
                            if row[j] - row[i] == d:
                                # Found a pair (i, j) in row r_idx
                                # Try to move j (or i)
                                # Let's move j
                                old_val = row[j]
                                low = row[j-1] + 1
                                high = row[j+1] - 1 if j < k else old_val + 10
                                
                                if high >= low:
                                    # Try random values
                                    # Bias towards values that don't create new conflicts
                                    # Just pick random
                                    new_val = random.randint(low, high)
                                    if new_val != old_val:
                                        # Simulate move
                                        removed_diffs = []
                                        for ii in range(j):
                                            removed_diffs.append(old_val - row[ii])
                                        for hh in range(j+1, k+1):
                                            removed_diffs.append(row[hh] - old_val)
                                            
                                        added_diffs = []
                                        for ii in range(j):
                                            added_diffs.append(new_val - row[ii])
                                        for hh in range(j+1, k+1):
                                            added_diffs.append(row[hh] - new_val)
                                            
                                        delta_pen = 0
                                        for dd in removed_diffs:
                                            c = curr_counts[dd]
                                            if c > 1:
                                                delta_pen -= 1
                                            if c == 2:
                                                curr_conflicts.discard(dd)
                                                
                                        for dd in added_diffs:
                                            c = curr_counts.get(dd, 0)
                                            if c >= 1:
                                                delta_pen += 1
                                            if c == 1:
                                                curr_conflicts.add(dd)
                                                
                                        if delta_pen < 0:
                                            # Commit
                                            for dd in removed_diffs:
                                                curr_counts[dd] -= 1
                                                if curr_counts[dd] == 0:
                                                    del curr_counts[dd]
                                                    
                                            for dd in added_diffs:
                                                if dd in curr_counts:
                                                    curr_counts[dd] += 1
                                                else:
                                                    curr_counts[dd] = 1
                                                    
                                            row[j] = new_val
                                            curr_penalty += delta_pen
                                            
                                            if new_val > curr_row_maxes[r_idx]:
                                                curr_row_maxes[r_idx] = new_val
                                            elif old_val == curr_row_maxes[r_idx]:
                                                curr_row_maxes[r_idx] = max(row)
                                                    
                                            curr_M = get_M_from_maxes(curr_row_maxes)
                                            if curr_penalty == 0 and curr_M < best_M:
                                                best_M = curr_M
                                                best_rows = copy.deepcopy(curr_rows)
                                                    
                                found = True
                                break
                    if found:
                        break
                if not found:
                    # This shouldn't happen if d is in conflicts
                    pass

    if best_rows is None:
        best_rows = greedy_init()
    return best_rows
