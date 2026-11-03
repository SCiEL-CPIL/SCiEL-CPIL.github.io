import random
import time

def construct(n_rows, k):
    # Target: n_rows rows, each with k+1 integers (indices 0..k)
    # Start with 0.
    
    # 1. Greedy Initialization with Randomization
    # We generate a few candidate initial solutions and pick the best one (lowest max)
    
    def greedy_init(randomize=False):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # To randomize, we can pick the next smallest available number, or a random available number within a range
            # Simple greedy: smallest available
            # Randomized greedy: smallest available + random noise?
            # Let's stick to standard greedy for the base, but we'll run SA from there.
            
            # Actually, a better init is to spread them out more.
            # Let's use a "random greedy" where we pick a random valid number from [last+1, last+limit]
            
            current_max = 0
            for _ in range(k):
                found = False
                # Try numbers from current_max + 1 upwards
                # To make it diverse, we can shuffle the order of checking? No, that breaks "increasing" property if we just append.
                # We must append increasing numbers.
                
                # Standard greedy:
                x = current_max + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                        current_max = x
                        found = True
                        break
                    x += 1
                if not found:
                    return None
            rows.append(row)
        return rows

    # Let's try to get a good initial solution
    # The seed program does greedy. Let's do the same but maybe try a few variations?
    # Actually, the greedy solution is often close to optimal for small k.
    # For k=7 (8 numbers), greedy is decent.
    
    init_rows = greedy_init()
    if not init_rows:
        # Fallback: random
        init_rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Pick random x > row[-1]
                x = row[-1] + random.randint(1, 10)
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
                else:
                    # If conflict, try again
                    pass
            # This loop might be infinite if bad luck. 
            # Better to use the deterministic greedy for init.
            pass
        
    # Re-init properly
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
                used.update(new_diffs)
        rows.append(row)
        
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    
    # 2. Simulated Annealing
    # State: rows
    # Move: Change one element in one row.
    
    # We need to efficiently check conflicts.
    # Maintain a global `used` set of all differences.
    # When we change a value in row r at index j from old_val to new_val:
    # 1. Remove all diffs involving old_val in row r from `used`.
    # 2. Check if new diffs involving new_val in row r conflict with `used`.
    # 3. If no conflict, add new diffs to `used`.
    
    # Note: `used` must contain ALL differences from ALL rows.
    # We must be careful not to remove a diff that is also present in another row.
    # So, `used` should be a Counter or we should check counts.
    # Since the requirement is "distinct over all rows", the count of each diff must be exactly 1.
    # So we can use a dictionary `diff_count` where `diff_count[d]` is the number of times diff d appears.
    # A move is valid if for all new diffs d, `diff_count[d]` (after removing old diffs) is 0.
    
    diff_count = {}
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                diff_count[d] = diff_count.get(d, 0) + 1
                
    # Verify initial state is valid (it should be from greedy)
    # max_count = max(diff_count.values())
    # assert max_count == 1
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    T = 100.0 # Initial temperature
    cooling_rate = 0.9995
    min_T = 1e-6
    
    current_max = max(max(r) for r in rows)
    best_max = current_max
    best_rows = [r[:] for r in rows]
    
    # Pre-compute row lengths
    row_len = k + 1
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random row and index
        r_idx = random.randint(0, n_rows - 1)
        # Index 0 is always 0. We can only change indices 1..k
        j_idx = random.randint(1, k)
        
        row = rows[r_idx]
        old_val = row[j_idx]
        
        # Generate a new value
        # Strategy: Try to move towards the center or randomize
        # Let's pick a random value in [row[j_idx-1]+1, row[j_idx+1]-1] if possible, or broader
        # To ensure we can find valid spots, we might need to jump.
        # Let's pick a new value randomly from a range around old_val.
        # Range: [max(1, old_val - 20), old_val + 20]
        
        # Ensure strict increase
        min_val = row[j_idx - 1] + 1
        max_val = row[j_idx + 1] - 1 if j_idx < k else old_val + 20
        
        if min_val > max_val:
            T *= cooling_rate
            continue
            
        new_val = random.randint(min_val, max_val)
        
        if new_val == old_val:
            T *= cooling_rate
            continue
            
        # Check validity of new_val in the row (distinctness)
        # Since we maintain sorted rows, and we only change one element,
        # we just need to ensure new_val is not equal to any other element in the row.
        # But wait, if we change row[j_idx], the row might not be sorted anymore?
        # No, we pick new_val between neighbors, so it stays in place and sorted.
        
        # Remove old diffs
        old_diffs_to_remove = []
        for i in range(row_len):
            if i == j_idx:
                continue
            d = abs(old_val - row[i])
            old_diffs_to_remove.append(d)
            
        # Tentatively remove from diff_count
        for d in old_diffs_to_remove:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
                
        # Check new diffs
        new_diffs_to_add = []
        conflict = False
        for i in range(row_len):
            if i == j_idx:
                continue
            d = abs(new_val - row[i])
            if d in diff_count:
                conflict = True
                break
            new_diffs_to_add.append(d)
            
        if conflict:
            # Restore
            for d in old_diffs_to_remove:
                diff_count[d] = diff_count.get(d, 0) + 1
            T *= cooling_rate
            continue
            
        # Apply move
        row[j_idx] = new_val
        for d in new_diffs_to_add:
            diff_count[d] = diff_count.get(d, 0) + 1
            
        # Calculate cost
        # Cost is primarily max value, but we must ensure no conflicts (which is guaranteed by check)
        # However, we might want to minimize the sum of squares to keep it compact?
        # Let's stick to minimizing max value.
        # But SA needs a smooth landscape. Min max is discontinuous.
        # Let's use Sum of Squares as the objective for SA, and track best max.
        
        # Calculate sum of squares for the changed row
        # Actually, calculating global sum of squares every step is O(N*k^2).
        # We can maintain `current_sum_sq`.
        
        # Let's just use Max Value as the objective, but accept moves that increase Max if they reduce "potential" conflicts?
        # No, we are in a valid state.
        # If we are in a valid state, any move that keeps it valid is an improvement if it reduces Max, or neutral if it keeps Max same.
        # If it increases Max, we should probably reject unless T is high.
        
        new_max = max(max(r) for r in rows)
        
        delta_max = new_max - current_max
        
        if delta_max <= 0:
            # Improvement or equal
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            # Worse max
            # Accept with probability exp(-delta_max / T)
            if random.random() < pow(2.718281828, -delta_max / T):
                current_max = new_max
            else:
                # Reject: Restore
                row[j_idx] = old_val
                for d in new_diffs_to_add:
                    diff_count[d] -= 1
                    if diff_count[d] == 0:
                        del diff_count[d]
                for d in old_diffs_to_remove:
                    diff_count[d] = diff_count.get(d, 0) + 1
                
        T *= cooling_rate
        
    return best_rows
