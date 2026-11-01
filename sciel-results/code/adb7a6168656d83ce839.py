import time
import random
import math

def construct(n_rows, k):
    """
    Constructs a set of n_rows difference triangles (rows of k+1 increasing integers starting at 0)
    such that all differences are distinct. Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_rows = None
    best_M = float('inf')
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
        
    def is_valid_move(rows, r, j, new_val, used_diffs):
        row = rows[r]
        old_val = row[j]
        
        # Check order constraints
        if j > 0 and new_val <= row[j-1]:
            return False
        if j < k and new_val >= row[j+1]:
            return False
            
        # Calculate differences to remove
        diffs_to_remove = set()
        for i in range(k + 1):
            if i == j:
                continue
            if i < j:
                diffs_to_remove.add(old_val - row[i])
            else:
                diffs_to_remove.add(row[i] - old_val)
                
        # Calculate differences to add
        diffs_to_add = set()
        for i in range(k + 1):
            if i == j:
                continue
            if i < j:
                diffs_to_add.add(new_val - row[i])
            else:
                diffs_to_add.add(row[i] - new_val)
                
        # Check for conflicts with existing used_diffs
        for d in diffs_to_add:
            if d in used_diffs and d not in diffs_to_remove:
                return False
                
        return True

    def apply_move(rows, r, j, new_val, used_diffs):
        row = rows[r]
        old_val = row[j]
        
        # Remove old diffs
        for i in range(k + 1):
            if i == j:
                continue
            if i < j:
                used_diffs.discard(old_val - row[i])
            else:
                used_diffs.discard(row[i] - old_val)
                
        # Update row
        row[j] = new_val
        
        # Add new diffs
        for i in range(k + 1):
            if i == j:
                continue
            if i < j:
                used_diffs.add(new_val - row[i])
            else:
                used_diffs.add(row[i] - new_val)

    def revert_move(rows, r, j, old_val, new_val, used_diffs):
        row = rows[r]
        
        # Remove new diffs (which are currently in used_diffs)
        for i in range(k + 1):
            if i == j:
                continue
            if i < j:
                used_diffs.discard(new_val - row[i])
            else:
                used_diffs.discard(row[i] - new_val)
                
        # Update row back
        row[j] = old_val
        
        # Add old diffs back
        for i in range(k + 1):
            if i == j:
                continue
            if i < j:
                used_diffs.add(old_val - row[i])
            else:
                used_diffs.add(row[i] - old_val)

    def greedy_random_init(rng):
        rows = []
        used_diffs = set()
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                candidates = []
                current_x = x
                # Find first 20 valid marks
                while len(candidates) < 20:
                    current_x += 1
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = current_x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        candidates.append(current_x)
                    if current_x > 300:
                        break
                        
                if not candidates:
                    # Fallback
                    while True:
                        x += 1
                        new_diffs = set()
                        valid = True
                        for a in row:
                            d = x - a
                            if d in used_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            candidates.append(x)
                            break
                            
                chosen = rng.choice(candidates)
                row.append(chosen)
                for a in row[:-1]:
                    used_diffs.add(chosen - a)
                
            rows.append(row)
            
        return rows, used_diffs

    rng = random.Random(42)
    
    # Run multiple iterations
    # We can increase the number of iterations if time permits
    max_inits = 20
    for init_idx in range(max_inits):
        if time.time() > deadline - 15:
            break
            
        rows, used_diffs = greedy_random_init(rng)
        current_M = get_max_entry(rows)
        
        if current_M < best_M:
            best_M = current_M
            best_rows = [row[:] for row in rows]
            
        # Local Search
        T = 5.0
        T_min = 0.01
        cooling = 0.995
        steps = 0
        max_steps = 20000
        
        while T > T_min and steps < max_steps:
            if time.time() > deadline - 5:
                break
                
            r = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            # Sample new value
            # Try to reduce M by sampling lower values more often
            # Uniform in [1, current_M + 5]
            max_val = current_M + 5
            new_val = rng.randint(1, max_val)
            
            if is_valid_move(rows, r, j, new_val, used_diffs):
                old_val = rows[r][j]
                apply_move(rows, r, j, new_val, used_diffs)
                
                new_M = get_max_entry(rows)
                delta = new_M - current_M
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_M = new_M
                    if current_M < best_M:
                        best_M = current_M
                        best_rows = [row[:] for row in rows]
                else:
                    revert_move(rows, r, j, old_val, new_val, used_diffs)
                    
                T *= cooling
                steps += 1
                
    if best_rows is None:
        # Fallback
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used_diffs:
                    row.append(x)
                    used_diffs |= new
            rows.append(row)
        return rows
        
    return best_rows
