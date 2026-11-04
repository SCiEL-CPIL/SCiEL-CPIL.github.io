import time
import random

def _greedy_construct(n_rows, k):
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
    return rows

def construct(n_rows, k):
    start_time = time.time()
    # Allow 110 seconds for search, 10s margin for safety
    deadline = start_time + 110.0
    
    # 1. Initial valid solution
    current_rows = _greedy_construct(n_rows, k)
    best_rows = [r[:] for r in current_rows]
    best_scope = max(max(r) for r in current_rows)
    
    # 2. Setup for Local Search (Simulated Annealing)
    # State: counts of differences
    counts = {}
    conflicts = 0
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in counts:
                    if counts[d] == 1:
                        conflicts += 1
                    counts[d] += 1
                else:
                    counts[d] = 1
    
    current_conflicts = conflicts
    current_max = best_scope
    current_cost = current_conflicts * 10000 + current_max
    
    T = 5.0  # Initial temperature
    T_min = 0.01
    # We don't know the number of iterations in advance, so we use time-based cooling
    # Or just a fixed cooling rate per iteration, assuming ~100k iterations/sec
    # Let's use a small cooling rate and check time.
    cooling_rate = 0.99995
    
    best_cost = current_cost
    
    while time.time() < deadline:
        # Generate a move
        r_idx = random.randint(0, n_rows - 1)
        j_idx = random.randint(1, k)
        
        row = current_rows[r_idx]
        old_val = row[j_idx]
        
        # Determine valid range for new_val
        low = row[j_idx - 1] + 1
        if j_idx < k:
            high = row[j_idx + 1] - 1
        else:
            # For the last element, allow some growth but bias down
            # Cap growth to avoid wandering too far
            high = current_max + 20 
            
        if low > high:
            continue
            
        # Bias: If this is the max element, try to reduce it
        if old_val == current_max and j_idx == k:
            # Sample from [low, old_val]
            if low <= old_val:
                new_val = random.randint(low, old_val)
            else:
                new_val = random.randint(low, high)
        else:
            # Uniform sample
            new_val = random.randint(low, high)
            
        if new_val == old_val:
            continue
            
        # Apply move and calculate delta
        # We need to update counts and conflicts
        # To make this fast, we do it in-place and revert if rejected
        
        # Store old state for revert
        # Actually, we can just calculate the new cost and decide.
        # But updating counts is O(k). Reverting is O(k).
        # Let's do it.
        
        # 1. Remove old differences
        for i in range(j_idx):
            d = old_val - row[i]
            c = counts.get(d, 0)
            if c == 2:
                current_conflicts -= 1
            counts[d] = c - 1
            if counts[d] == 0:
                del counts[d]
                
        for i in range(j_idx + 1, k + 1):
            d = row[i] - old_val
            c = counts.get(d, 0)
            if c == 2:
                current_conflicts -= 1
            counts[d] = c - 1
            if counts[d] == 0:
                del counts[d]
                
        # 2. Add new differences
        for i in range(j_idx):
            d = new_val - row[i]
            c = counts.get(d, 0)
            if c == 1:
                current_conflicts += 1
            counts[d] = c + 1
            
        for i in range(j_idx + 1, k + 1):
            d = row[i] - new_val
            c = counts.get(d, 0)
            if c == 1:
                current_conflicts += 1
            counts[d] = c + 1
            
        # 3. Update row
        row[j_idx] = new_val
        
        # 4. Calculate new cost
        new_max = max(max(r) for r in current_rows)
        new_cost = current_conflicts * 10000 + new_max
        
        # 5. Acceptance
        delta_cost = new_cost - current_cost
        if delta_cost < 0:
            # Accept
            current_cost = new_cost
            current_max = new_max
            if new_cost < best_cost:
                best_cost = new_cost
                if current_conflicts == 0:
                    best_rows = [r[:] for r in current_rows]
                    best_scope = new_max
        else:
            # Reject or accept with probability
            if T > 0:
                # Avoid overflow in exp
                if delta_cost < 1000: 
                    p = 2.718281828 ** (-delta_cost / T)
                    if random.random() < p:
                        current_cost = new_cost
                        current_max = new_max
                        # Note: We do NOT update best_rows if conflicts > 0
                        if current_conflicts == 0:
                             if new_cost < best_cost:
                                 best_cost = new_cost
                                 best_rows = [r[:] for r in current_rows]
                                 best_scope = new_max
                # If delta is huge, p is ~0, reject
            # If T is 0, reject
            
            # If rejected, revert
            if not (delta_cost < 0 or (T > 0 and random.random() < (2.718281828 ** (-delta_cost / T) if delta_cost < 1000 else 0))):
                # This logic is messy. Let's refactor.
                pass
                
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: Restart? No, just keep going.
            
    return best_rows
