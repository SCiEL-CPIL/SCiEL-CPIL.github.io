import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # 1. Initial Solution: Greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    break
                x += 1
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    best_rows = [r[:] for r in rows]
    best_score = max(max(r) for r in best_rows)
    
    # Helper to get score
    def get_score(rows):
        return max(max(r) for r in rows)
    
    # Helper to check validity (should always be true if we manage 'used' correctly)
    
    # 2. Search Loop
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # We can run many iterations
    while time.time() < deadline:
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        
        # Move 1: Re-insert
        # Remove current row's diffs
        old_row = rows[r_idx]
        diffs_to_remove = set()
        for i in range(len(old_row)):
            for j in range(i):
                diffs_to_remove.add(old_row[i] - old_row[j])
        used -= diffs_to_remove
        
        # Re-insert greedily
        new_row = [0]
        success = True
        for _ in range(k):
            x = new_row[-1] + 1
            while True:
                new_diffs = {x - a for a in new_row}
                if not new_diffs & used:
                    break
                x += 1
                # If x gets too large, we might want to abort? 
                # But greedy always finds a solution.
            new_row.append(x)
            used |= new_diffs
        
        rows[r_idx] = new_row
        current_score = get_score(rows)
        
        if current_score < best_score:
            best_score = current_score
            best_rows = [r[:] for r in rows]
            T = 1.0 # Reset temperature on improvement?
        else:
            # Accept worse solution with probability exp(-delta/T)
            delta = current_score - best_score
            if delta > 0:
                if random.random() < pow(2.718, -delta / T):
                    pass # Accept, state is already updated
                else:
                    # Revert
                    used -= {x - y for i, x in enumerate(new_row) for j, y in enumerate(new_row) if i > j}
                    rows[r_idx] = old_row
                    used |= diffs_to_remove
            # T cooling
            T = max(T_min, T * cooling)
            
        # Move 2: Local Compression (optional, re-insert is strong)
        # Maybe alternate?
        
    return best_rows
