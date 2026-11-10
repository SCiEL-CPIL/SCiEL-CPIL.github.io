import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Greedy initialization
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Try to extend the row
                # We can try to find the smallest next x such that no diff is used
                # Simple greedy: try x = current_max + 1, +2, ...
                # But we can be smarter: try to fit in a small range.
                # For initialization, simple greedy is fine.
                x = current_max + 1
                found = False
                while True:
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        current_max = x
                        found = True
                        break
                    x += 1
                if not found:
                    # Should not happen for reasonable n_rows, k
                    break
            rows.append(row)
        return rows

    rows = greedy_init()
    
    def get_cost(rows):
        used = set()
        conflicts = 0
        max_val = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
                if r[i] > max_val:
                    max_val = r[i]
        # Penalty for conflicts is high to prioritize validity
        return conflicts * 10000 + max_val, conflicts, max_val

    current_rows = [r[:] for r in rows]
    current_cost, current_conflicts, current_max = get_cost(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_cost = current_cost
    best_conflicts = current_conflicts
    best_max = current_max
    
    if best_conflicts == 0:
        # If greedy already found a valid solution, we can try to improve scope
        # But usually greedy is not optimal.
        pass

    T = 500.0
    cooling_rate = 0.995
    min_T = 0.01
    
    # Precompute indices for moves
    # A move is (row_idx, elem_idx) where elem_idx is 1..k (0 is fixed at 0)
    
    while time.time() < deadline and T > min_T:
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        e_idx = random.randint(1, k) # 1 to k
        
        old_val = current_rows[r_idx][e_idx]
        
        # Determine range for new value
        # Must be > current_rows[r_idx][e_idx-1] and < current_rows[r_idx][e_idx+1] (if exists)
        lower = current_rows[r_idx][e_idx - 1] + 1
        upper = current_rows[r_idx][e_idx + 1] - 1 if e_idx < k else current_max + 100 # Allow expansion
        
        if lower >= upper:
            T *= cooling_rate
            continue
            
        # Propose a new value
        # Uniform in [lower, upper] might be too wide.
        # Let's try a local move: old_val + delta
        # delta in [-10, 10]
        delta = random.randint(-10, 10)
        new_val = old_val + delta
        
        if new_val < lower or new_val > upper:
            T *= cooling_rate
            continue
            
        # Apply move
        current_rows[r_idx][e_idx] = new_val
        
        new_cost, new_conflicts, new_max = get_cost(current_rows)
        delta_cost = new_cost - current_cost
        
        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
            current_cost = new_cost
            current_conflicts = new_conflicts
            current_max = new_max
            
            if new_conflicts < best_conflicts or (new_conflicts == best_conflicts and new_cost < best_cost):
                best_conflicts = new_conflicts
                best_cost = new_cost
                best_max = new_max
                best_rows = [r[:] for r in current_rows]
                
        else:
            # Revert
            current_rows[r_idx][e_idx] = old_val
            
        T *= cooling_rate
        
        # If we have a valid solution, we can try to lower T faster or just keep going
        # to squeeze out more scope reduction.
        
    # Return the best valid solution found
    if best_conflicts == 0:
        return best_rows
    else:
        # If no valid solution found, return the one with fewest conflicts?
        # The task requires a valid result. If we can't find one, we should return the greedy one if it was valid?
        # Greedy is always valid by construction?
        # Let's check if greedy was valid.
        g_cost, g_conf, g_max = get_cost(rows)
        if g_conf == 0:
            return rows
        # If even greedy is invalid (shouldn't happen), return best attempt
        return best_rows
