import random
import copy

def construct(n_rows, k):
    # Build initial solution using greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    # Compute initial max
    max_val = max(max(r) for r in rows)
    
    # Simulated annealing to reduce max
    # We'll try to move individual non-zero elements to smaller positions
    
    # Precompute: for each row, the differences involving each element
    # We'll maintain used_diffs as a set
    
    used_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    best_rows = copy.deepcopy(rows)
    best_max = max_val
    
    # SA parameters
    T = 50.0
    cooling = 0.999
    iterations = 50000
    
    random.seed(42)
    
    for it in range(iterations):
        # Pick a random row and a random non-first element
        ri = random.randint(0, n_rows - 1)
        ei = random.randint(1, k)  # element index (0 is always 0)
        
        row = rows[ri]
        old_val = row[ei]
        
        # Remove old differences
        old_diffs = set()
        for j in range(len(row)):
            if j != ei:
                d = abs(row[ei] - row[j])
                old_diffs.add(d)
        
        # Try to find a smaller position for this element
        # The new value must keep the row strictly increasing
        # If ei > 0: new_val > row[ei-1]
        # If ei < k: new_val < row[ei+1]
        
        lower = row[ei-1] + 1 if ei > 0 else 1
        upper = row[ei+1] - 1 if ei < k else max_val
        
        # Try random smaller values in [lower, old_val-1]
        if old_val <= lower:
            T *= cooling
            continue
            
        candidates = list(range(lower, old_val))
        if not candidates:
            T *= cooling
            continue
            
        # Try a few random candidates
        for _ in range(3):
            new_val = random.choice(candidates)
            
            # Check if new differences are valid
            new_diffs = set()
            valid = True
            for j in range(len(row)):
                if j != ei:
                    d = abs(new_val - row[j])
                    if d in used_diffs and d not in old_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
            
            if not valid:
                continue
            
            # Also need to ensure new_diffs don't have internal conflicts (they shouldn't since they're from one row)
            # But we need to check new_diffs against (used_diffs - old_diffs)
            
            # Apply move
            row[ei] = new_val
            used_diffs -= old_diffs
            used_diffs |= new_diffs
            
            new_max = max(max(r) for r in rows)
            
            if new_max < best_max:
                best_max = new_max
                best_rows = copy.deepcopy(rows)
            
            # Accept or reject based on SA
            # Energy = new_max (we want to minimize)
            # Since we're only trying smaller values, new_max <= old_max always
            # So always accept
            # Actually, moving to a smaller value can't increase the max, so always accept
            
        T *= cooling
    
    return best_rows
