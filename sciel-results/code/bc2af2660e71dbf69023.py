import time
import random
import sys

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    start_time = time.time()
    deadline = 140.0

    def is_row_valid(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def greedy_construct():
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
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    # Run multiple restarts to find a good solution
    best_rows = None
    best_max = float('inf')
    
    while time.time() - start_time < deadline:
        rows, used_diffs = greedy_construct()
        current_max = max(max(r) for r in rows)
        
        if current_max < best_max:
            best_max = current_max
            best_rows = [r[:] for r in rows]
        
        # Local Search with SA
        temp = 50.0
        cool_rate = 0.995
        min_temp = 0.5
        no_improve_count = 0
        
        while time.time() - start_time < deadline and temp > min_temp:
            r_idx = random.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            
            # Pick position
            pos = random.randint(1, k)
            
            lower = old_row[pos-1] + 1
            upper = old_row[pos+1] - 1 if pos < k else current_max + 20
            
            if lower >= upper:
                temp *= cool_rate
                continue
                
            # Try to find a valid move
            # Bias towards smaller values if we are trying to reduce max
            # If pos == k and old_row[-1] == current_max, try to shrink
            if pos == k and old_row[-1] == current_max:
                # Try values in [lower, current_max - 1]
                if lower < current_max:
                    new_val = random.randint(lower, current_max - 1)
                else:
                    new_val = random.randint(lower, upper)
            else:
                new_val = random.randint(lower, upper)
            
            new_row = old_row[:]
            new_row[pos] = new_val
            
            if not is_row_valid(new_row):
                temp *= cool_rate
                continue
                
            new_diffs = get_row_diffs(new_row)
            old_diffs = get_row_diffs(old_row)
            
            foreign_used = used_diffs - old_diffs
            if new_diffs & foreign_used:
                temp *= cool_rate
                continue
            
            # Valid move
            # Calculate new max
            # To speed up, if we didn't change the max element, max stays same or decreases?
            # If we increased a non-max element, max might increase.
            # Let's just compute it.
            new_max_val = max(max(r) for r in rows) # Placeholder, need to update row first
            
            # Update state
            rows[r_idx] = new_row
            used_diffs.difference_update(old_diffs)
            used_diffs.update(new_diffs)
            
            # Recalculate max properly
            # We can optimize by keeping a sorted list of maxes?
            # For now, simple calculation.
            new_max_val = max(max(r) for r in rows)
            
            delta = new_max_val - current_max
            
            if delta < 0:
                current_max = new_max_val
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Accept with prob
                if random.random() < pow(2.71828, -delta / temp):
                    current_max = new_max_val
                    no_improve_count += 1
                else:
                    # Revert
                    rows[r_idx] = old_row
                    used_diffs.difference_update(new_diffs)
                    used_diffs.update(old_diffs)
                    no_improve_count += 1
            
            if no_improve_count > 1000:
                # Restart local search or break
                break
                
            temp *= cool_rate

    return best_rows
