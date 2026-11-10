import sys
import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    # Improved Greedy Initialization with Randomization
    def greedy_init(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Start from a bit higher than the immediate next to allow some variation?
                # No, greedy should be tight. We add randomness in the SA.
                x = current_max + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used.update(new_diffs)
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows

    # State
    best_rows = greedy_init(n_rows, k)
    current_rows = [list(r) for r in best_rows]
    
    # Precompute difference counts
    diff_counts = defaultdict(int)
    conflict_count = 0
    
    def add_diff(d):
        nonlocal conflict_count
        old_c = diff_counts[d]
        if old_c == 1:
            conflict_count += 1
        diff_counts[d] = old_c + 1

    def remove_diff(d):
        nonlocal conflict_count
        old_c = diff_counts[d]
        if old_c == 2:
            conflict_count -= 1
        diff_counts[d] = old_c - 1
        if diff_counts[d] == 0:
            del diff_counts[d]

    # Initialize counts
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                add_diff(d)
                
    current_scope = max(max(row) for row in current_rows)
    
    def get_cost(scope, conflicts):
        if conflicts == 0:
            return scope
        return 10000 + conflicts

    current_cost = get_cost(current_scope, conflict_count)
    best_cost = current_cost
    best_rows_found = [list(r) for r in current_rows]
    
    start_time = time.time()
    time_limit = 115.0
    
    # SA Parameters
    T = 20.0
    cooling_rate = 0.99995
    min_T = 0.05
    
    # Restart logic
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit:
        # Restart if stuck
        if no_improve_steps > max_no_improve:
            # Perturb best solution to restart
            current_rows = [list(r) for r in best_rows_found]
            # Randomly swap some elements or shift
            for _ in range(5):
                r = random.randint(0, n_rows - 1)
                i = random.randint(1, k)
                # Shift by small random amount
                delta = random.randint(-5, 5)
                old_val = current_rows[r][i]
                new_val = old_val + delta
                # Check bounds roughly
                min_v = current_rows[r][i-1] + 1
                max_v = current_rows[r][i+1] - 1 if i < k else current_scope + 20
                if min_v <= new_val <= max_v:
                    current_rows[r][i] = new_val
            # Recalculate state
            diff_counts.clear()
            conflict_count = 0
            for row in current_rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        add_diff(d)
            current_scope = max(max(row) for row in current_rows)
            current_cost = get_cost(current_scope, conflict_count)
            no_improve_steps = 0
            T = 20.0 # Reset temperature
            
        r = random.randint(0, n_rows - 1)
        i = random.randint(1, k)
        
        row = current_rows[r]
        old_val = row[i]
        
        # Determine range
        min_val = row[i-1] + 1
        max_val = row[i+1] - 1 if i < k else current_scope + 20
        
        if min_val > max_val:
            continue
            
        # Random perturbation: Uniform in range, but biased towards smaller changes?
        # Let's try a mix: 50% small step, 50% uniform
        if random.random() < 0.5:
            step = random.randint(1, 5)
            if random.random() < 0.5:
                new_val = old_val + step
            else:
                new_val = old_val - step
            if new_val < min_val: new_val = min_val
            if new_val > max_val: new_val = max_val
        else:
            new_val = random.randint(min_val, max_val)
            
        if new_val == old_val:
            continue
            
        # Calculate deltas
        # Remove old diffs
        for j in range(len(row)):
            if j == i: continue
            if j < i:
                d = old_val - row[j]
            else:
                d = row[j] - old_val
            remove_diff(d)
                
        # Add new diffs
        for j in range(len(row)):
            if j == i: continue
            if j < i:
                d = new_val - row[j]
            else:
                d = row[j] - new_val
            add_diff(d)
                
        # Update row
        row[i] = new_val
        
        # Calculate new scope
        # Only row r changed.
        new_row_max = max(row)
        # We need max of all rows.
        # Since n_rows is small, just compute it.
        new_scope = max(max(rw) for rw in current_rows)
        
        new_cost = get_cost(new_scope, conflict_count)
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            # Accept
            current_scope = new_scope
            current_cost = new_cost
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows_found = [list(rw) for rw in current_rows]
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            # Revert
            # Remove new diffs
            for j in range(len(row)):
                if j == i: continue
                if j < i:
                    d = new_val - row[j]
                else:
                    d = row[j] - new_val
                remove_diff(d)
                    
            # Add old diffs
            for j in range(len(row)):
                if j == i: continue
                if j < i:
                    d = old_val - row[j]
                else:
                    d = row[j] - old_val
                add_diff(d)
                    
            row[i] = old_val
            
        T *= cooling_rate
        if T < min_T:
            T = min_T

    return best_rows_found
