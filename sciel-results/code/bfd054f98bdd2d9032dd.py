import random
import time

def construct(n_rows, k):
    # Deadline: 150s total, leave 10s for cleanup -> 140s for search
    deadline = time.time() + 140.0
    
    # 1. Initial Solution: Greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
        
    def get_cost(rows):
        # Prioritize minimizing the maximum entry
        max_val = max(max(r) for r in rows)
        sum_val = sum(sum(r) for r in rows)
        # Weight max heavily
        return max_val * 1000 + sum_val

    best_rows = [r[:] for r in rows]
    best_cost = get_cost(best_rows)
    current_cost = best_cost
    
    # Rebuild used set for initial state (already done in greedy, but let's be sure)
    # The greedy loop above already populates 'used' correctly.
    
    def is_valid_move(row, idx, new_val):
        # Check if changing row[idx] to new_val is valid
        # 1. Strictly increasing
        if idx > 0 and new_val <= row[idx-1]:
            return False
        if idx < len(row) - 1 and new_val >= row[idx+1]:
            return False
            
        old_val = row[idx]
        if old_val == new_val:
            return True
            
        # Calculate old diffs
        old_diffs = set()
        for j in range(len(row)):
            if j == idx:
                continue
            if j < idx:
                old_diffs.add(old_val - row[j])
            else:
                old_diffs.add(row[j] - old_val)
                
        # Calculate new diffs
        new_diffs = set()
        for j in range(len(row)):
            if j == idx:
                continue
            if j < idx:
                new_diffs.add(new_val - row[j])
            else:
                new_diffs.add(row[j] - new_val)
                
        # Check for internal duplicates in new_diffs
        # Since we use a set, duplicates are collapsed. 
        # We need to verify if the *list* of differences had duplicates.
        # If it did, the set size will be smaller than the count (k).
        # Count of diffs is k (since we compare idx with all other k elements).
        if len(new_diffs) != k:
            return False
            
        # Check collision with global used
        # The new global set would be (used - old_diffs) U new_diffs
        # Collision if new_diffs intersects (used - old_diffs)
        if new_diffs & (used - old_diffs):
            return False
             
        return True

    def apply_move(row, idx, new_val):
        old_val = row[idx]
        if old_val == new_val:
            return
        
        # Remove old diffs
        old_diffs = set()
        for j in range(len(row)):
            if j == idx: continue
            if j < idx:
                old_diffs.add(old_val - row[j])
            else:
                old_diffs.add(row[j] - old_val)
        used -= old_diffs
        
        row[idx] = new_val
        
        # Add new diffs
        new_diffs = set()
        for j in range(len(row)):
            if j == idx: continue
            if j < idx:
                new_diffs.add(new_val - row[j])
            else:
                new_diffs.add(row[j] - new_val)
        used |= new_diffs

    # Simulated Annealing
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        if T < T_min:
            # Restart with higher T if stuck? Or just stop.
            # Let's restart a few times if time permits
            T = 10.0 
            if time.time() > deadline - 5:
                break

        # Generate Move
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        pos = random.randint(1, k) # 1 to k
        
        current_val = row[pos]
        low = row[pos-1] + 1
        
        # Strategy: 
        # 60% chance to try to decrease (shrink scope)
        # 40% chance to try to increase/shuffle (escape local minima)
        
        if current_val > low and random.random() < 0.6:
            # Pick a value between low and current_val - 1
            # Bias towards smaller values?
            new_val = random.randint(low, current_val - 1)
        else:
            # Pick a value >= current_val
            if pos < k:
                high = row[pos+1] - 1
            else:
                # For the last element, allow some expansion to escape traps
                high = current_val + 20
            
            if high < current_val:
                high = current_val
            new_val = random.randint(current_val, high)
            
        if new_val == current_val:
            continue
            
        if is_valid_move(row, pos, new_val):
            new_cost = get_cost(rows)
            delta = new_cost - current_cost
            
            # Accept if better or with probability based on T
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                apply_move(row, pos, new_val)
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_rows = [r[:] for r in rows]
            
        T *= cooling_rate
        
    return best_rows
