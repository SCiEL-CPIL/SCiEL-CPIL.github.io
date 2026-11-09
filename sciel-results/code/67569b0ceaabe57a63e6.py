import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate an initial feasible solution using the greedy method
    # This provides a valid upper bound (scope) to start from
    def greedy_initial():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
            rows.append(row)
        return rows

    initial_rows = greedy_initial()
    best_rows = [r[:] for r in initial_rows]
    best_scope = max(max(r) for r in best_rows)
    
    # Flatten rows into a single list for easier manipulation
    # Structure: [row0_1, ..., row0_k, row1_1, ..., row1_k, ...]
    # Note: Index 0 of each row is always 0, so we only vary indices 1..k
    def flatten(rows):
        flat = []
        for r in rows:
            flat.extend(r[1:]) # Skip the leading 0
        return flat

    def unflatten(flat):
        rows = []
        idx = 0
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                row.append(flat[idx])
                idx += 1
            rows.append(row)
        return rows

    current_flat = flatten(initial_rows)
    current_scope = best_scope
    
    # Helper to calculate collisions and max scope efficiently
    def evaluate(flat):
        rows = unflatten(flat)
        used = set()
        collisions = 0
        max_val = 0
        for r in rows:
            # Check strictly increasing
            if any(r[i] >= r[i+1] for i in range(len(r)-1)):
                return float('inf'), float('inf')
            
            max_val = max(max_val, max(r))
            
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        collisions += 1
                    else:
                        used.add(d)
        return max_val, collisions

    # Simulated Annealing
    T = 1000.0
    cooling = 0.9995
    
    # Pre-calculate total elements to iterate
    n_elements = n_rows * k
    
    while time.time() < deadline:
        if T < 0.1:
            break
            
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Shift an element
            idx = random.randrange(n_elements)
            delta = random.choice([-2, -1, 1, 2])
            new_flat = current_flat[:]
            new_flat[idx] += delta
            
            # Ensure strictly increasing within the row
            row_idx = idx // k
            col_idx = idx % k
            
            # Check bounds and monotonicity for the modified row
            row = unflatten(new_flat)[row_idx]
            if row[col_idx] < 0:
                continue
            if col_idx > 0 and row[col_idx] <= row[col_idx-1]:
                continue
            if col_idx < k-1 and row[col_idx] >= row[col_idx+1]:
                continue
                
        elif move_type < 0.8:
            # Swap two elements (possibly in different rows)
            idx1 = random.randrange(n_elements)
            idx2 = random.randrange(n_elements)
            if idx1 == idx2:
                continue
            new_flat = current_flat[:]
            new_flat[idx1], new_flat[idx2] = new_flat[idx2], new_flat[idx1]
            
            # Check monotonicity for affected rows
            r1 = idx1 // k
            r2 = idx2 // k
            rows = unflatten(new_flat)
            
            valid = True
            if any(rows[r1][i] >= rows[r1][i+1] for i in range(len(rows[r1])-1)):
                valid = False
            if r1 != r2 and any(rows[r2][i] >= rows[r2][i+1] for i in range(len(rows[r2])-1)):
                valid = False
            if not valid:
                continue
                
        else:
            # Local random restart of a single row (keep other rows fixed)
            # This helps escape local minima where a specific row is "stuck"
            row_idx = random.randrange(n_rows)
            new_flat = current_flat[:]
            
            # Regenerate this row greedily but with some randomness or just reset to 0..k?
            # Let's try a small perturbation: shift the whole row by a small amount
            # Or just randomize the row entirely? Randomizing might break feasibility badly.
            # Let's do a "random walk" on this specific row
            for c in range(k):
                idx = row_idx * k + c
                new_flat[idx] += random.randint(-5, 5)
            
            # Check monotonicity
            rows = unflatten(new_flat)
            if any(rows[row_idx][i] >= rows[row_idx][i+1] for i in range(len(rows[row_idx])-1)):
                continue
            if any(v < 0 for v in rows[row_idx]):
                continue

        new_scope, new_collisions = evaluate(new_flat)
        old_scope, old_collisions = evaluate(current_flat)
        
        # Objective: Minimize Scope. If Scope is same, minimize Collisions.
        # Actually, we must maintain 0 collisions for validity.
        # Let's define cost = scope * 1000 + collisions. 
        # But we want to reduce scope.
        # If new_collisions > 0, it's invalid. We should avoid invalid states unless T is high?
        # Better strategy: Only accept valid states, or use a penalty.
        # Given the strict constraint, let's stick to valid states only for simplicity and robustness,
        # but allow "collisions" if we add a heavy penalty and hope to jump out?
        # No, the problem requires distinct differences. An invalid solution scores 0.
        # So we must only return valid solutions.
        # We can explore invalid states if we ensure we don't *return* them, but SA usually works on a cost function.
        # Let's define cost = scope + collisions * 10000.
        # We accept worse costs with probability exp(-delta_cost / T).
        
        old_cost = old_scope + old_collisions * 10000
        new_cost = new_scope + new_collisions * 10000
        
        delta_cost = new_cost - old_cost
        
        if delta_cost < 0 or (new_collisions == 0 and random.random() < pow(2.71828, -delta_cost / T)):
            current_flat = new_flat
            current_scope = new_scope
            
            if new_collisions == 0:
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = unflatten(new_flat)
        
        T *= cooling

    return best_rows
# EVOLVE-BLOCK-END
