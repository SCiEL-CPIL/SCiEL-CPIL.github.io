import time
import sys
import random

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 elements.
    Returns a list of rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Total number of differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound for M (max element)
    lower_bound = max(total_diffs, k)
    
    def get_differences(row):
        return [row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))]

    def count_conflicts(rows):
        diffs = []
        for r in rows:
            diffs.extend(get_differences(r))
        # Count duplicates
        from collections import Counter
        counts = Counter(diffs)
        conflicts = 0
        for count in counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    def is_valid(rows):
        diffs = []
        for r in rows:
            diffs.extend(get_differences(r))
        return len(diffs) == len(set(diffs))

    # Initial solution: Greedy
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
                    used |= new_diffs
            rows.append(row)
        return rows

    current_rows = greedy_construct()
    best_rows = [r[:] for r in current_rows]
    best_score = count_conflicts(best_rows)
    best_max = max(max(r) for r in best_rows)
    
    # If greedy is already valid, we might still try to reduce the max value?
    # The task asks for "largest entry as small as possible".
    # Greedy tends to produce large values.
    # We can use SA to minimize the max value while keeping conflicts at 0.
    
    # Objective: Minimize (conflicts * 1000000 + max_value)
    # Or just minimize conflicts first, then max value.
    
    def objective(rows):
        conflicts = count_conflicts(rows)
        max_val = max(max(r) for r in rows)
        return conflicts * 1000000 + max_val

    current_obj = objective(current_rows)
    best_obj = current_obj
    
    # Simulated Annealing
    # We need to define moves.
    # Move: Pick a random row, pick a random element (not 0), change its value.
    # Or: Pick a random row, regenerate it using greedy with a different starting offset or random choices?
    # Better: Local move. Change one element in one row.
    
    temp = 1000.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Generate neighbor
        new_rows = [r[:] for r in current_rows]
        row_idx = random.randint(0, n_rows - 1)
        # Pick an element index > 0 to change
        elem_idx = random.randint(1, k)
        
        # Try a few random new values for this element
        # Keep the row sorted?
        # If we change row[row_idx], we must ensure it stays between row[row_idx-1] and row[row_idx+1]
        # And strictly increasing.
        
        low = new_rows[row_idx][elem_idx - 1] + 1
        high = new_rows[row_idx][elem_idx + 1] - 1 if elem_idx < k else best_max + 10
        
        if low > high:
            # Try swapping or shifting? 
            # For simplicity, if range is invalid, skip this move or try a different element
            continue
            
        new_val = random.randint(low, high)
        
        # If new_val is same as old, skip
        if new_val == new_rows[row_idx][elem_idx]:
            continue
            
        new_rows[row_idx][elem_idx] = new_val
        
        # Check if row is still strictly increasing
        if not all(new_rows[row_idx][i] < new_rows[row_idx][i+1] for i in range(k)):
            continue
            
        new_obj = objective(new_rows)
        
        delta = new_obj - current_obj
        
        if delta < 0 or random.random() < pow(2.718, -delta / temp):
            current_rows = new_rows
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_rows = [r[:] for r in current_rows]
                
        temp *= cooling
        
        # If we found a valid solution (conflicts=0), we can try to just minimize max value
        # The objective function handles this.
        
        # If best_obj has 0 conflicts, we are in a valid state.
        # We can stop if time is running out.
        
    return best_rows
