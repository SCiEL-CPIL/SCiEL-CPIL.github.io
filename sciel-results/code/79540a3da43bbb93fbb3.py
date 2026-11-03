import random
import math

def _greedy_init(n_rows, k, seed=0):
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Try a few random steps to avoid getting stuck in a bad local minimum of the greedy
            # Actually, standard greedy is deterministic. Let's just do standard greedy.
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _get_max(rows):
    return max(max(r) for r in rows)

def _is_valid_move(rows, r_idx, j_idx, new_val, used):
    # Check order
    row = rows[r_idx]
    if j_idx > 0 and new_val <= row[j_idx - 1]:
        return False
    if j_idx < len(row) - 1 and new_val >= row[j_idx + 1]:
        return False
    
    old_val = row[j_idx]
    
    # Calculate removed and added diffs
    removed = set()
    added = set()
    for i in range(len(row)):
        if i == j_idx:
            continue
        removed.add(abs(old_val - row[i]))
        added.add(abs(new_val - row[i]))
        
    # Check conflicts
    # A diff is a conflict if it is in 'added', NOT in 'removed', and IS in 'used'
    # Note: 'used' contains all diffs from all rows.
    # If a diff is in 'removed', it means it was present in 'used' due to this specific change's origin, 
    # so it's "free" to be reused if it also appears in 'added'.
    # Wait, if d is in removed and added, it's fine.
    # If d is in added and not in removed, it must not be in used.
    for d in added:
        if d in removed:
            continue
        if d in used:
            return False
    return True

def _apply_move(rows, r_idx, j_idx, new_val, used):
    row = rows[r_idx]
    old_val = row[j_idx]
    
    removed = set()
    added = set()
    for i in range(len(row)):
        if i == j_idx:
            continue
        removed.add(abs(old_val - row[i]))
        added.add(abs(new_val - row[i]))
        
    # Update used
    # We need to be careful. 
    # used currently contains all diffs.
    # We remove 'removed' and add 'added'.
    # But if a diff is in both, net effect is 0.
    # If we do used -= removed; used |= added, it works.
    used -= removed
    used |= added
    
    row[j_idx] = new_val

def _undo_move(rows, r_idx, j_idx, new_val, old_val, used):
    row = rows[r_idx]
    
    removed = set()
    added = set()
    for i in range(len(row)):
        if i == j_idx:
            continue
        # Current value is new_val. We are going back to old_val.
        # So "removed" from current state are diffs with new_val.
        # "added" to current state are diffs with old_val.
        removed.add(abs(new_val - row[i]))
        added.add(abs(old_val - row[i]))
        
    used -= removed
    used |= added
    
    row[j_idx] = old_val

def construct(n_rows, k):
    # 1. Generate initial solutions
    best_rows = None
    best_max = float('inf')
    
    # Try a few different seeds for randomization in greedy if we had it, 
    # but standard greedy is deterministic. 
    # Let's just run the standard greedy once.
    rows = _greedy_init(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_max = _get_max(rows)
    
    # 2. Simulated Annealing
    # Build initial 'used' set
    used = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                used.add(r[j] - r[i])
                
    current_max = best_max
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    # Run for a fixed number of steps or until T is very low
    steps = 50000
    for step in range(steps):
        T *= cooling
        if T < T_min:
            break
            
        # Pick random row and index
        r_idx = random.randint(0, n_rows - 1)
        j_idx = random.randint(1, k) # Index 0 is always 0
        
        row = rows[r_idx]
        old_val = row[j_idx]
        
        # Propose new value
        # Try to move towards smaller values to minimize max, but allow some exploration
        # Range: [row[j_idx-1]+1, row[j_idx+1]-1]
        min_val = row[j_idx-1] + 1
        max_val = row[j_idx+1] - 1 if j_idx < k else old_val + 10 # If last, allow increase? No, we want to minimize.
        # Actually, if we are minimizing max, we rarely want to increase. 
        # But to escape local minima, we might need to.
        # Let's try a random value in a window around old_val.
        # Window size can shrink with T.
        window = max(1, int(T * 5))
        new_val = random.randint(max(min_val, old_val - window), min(max_val, old_val + window))
        
        if new_val == old_val:
            continue
            
        if not _is_valid_move(rows, r_idx, j_idx, new_val, used):
            continue
            
        # Calculate new max
        # Only the max of this row might change, or the global max.
        # Efficiently:
        # If new_val is the max of this row and it decreases, global max might decrease.
        # If new_val increases and it was the max, global max might increase.
        # Simplest: just compute max of the row and compare with global max of other rows?
        # Too slow. 
        # Let's just compute the new global max.
        # Since k is small (8), and n_rows is small (4), this is fast.
        # But we can be smarter.
        
        # Apply move
        _apply_move(rows, r_idx, j_idx, new_val, used)
        new_max = _get_max(rows)
        
        # Acceptance
        if new_max < current_max:
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            # Accept with probability
            p = math.exp(-(new_max - current_max) / T)
            if random.random() < p:
                current_max = new_max
            else:
                # Undo
                _undo_move(rows, r_idx, j_idx, new_val, old_val, used)
                
    return best_rows
