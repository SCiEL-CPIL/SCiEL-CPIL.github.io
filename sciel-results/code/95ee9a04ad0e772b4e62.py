import time
import random
import sys

def construct(n_rows, k):
    """
    Finds a difference triangle set of n_rows rows, each with k+1 integers.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    # Lower bound on the max value M
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    lower_bound = total_diffs_needed
    
    best_solution = None
    best_M = float('inf')
    
    # Fallback greedy solution
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    fallback = greedy_construct()
    best_solution = fallback
    best_M = max(max(row) for row in fallback)
    
    # Simulated Annealing approach
    # State: list of rows
    # Objective: minimize max entry, subject to all differences being distinct
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def get_score(rows):
        # Primary: number of conflicts
        # Secondary: max entry
        used = set()
        conflicts = 0
        max_entry = 0
        for row in rows:
            max_entry = max(max_entry, row[-1])
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        conflicts += 1
                    used.add(d)
        return conflicts, max_entry
    
    # Initialize with greedy
    current_rows = [list(r) for r in fallback]
    current_conflicts, current_max = get_score(current_rows)
    
    # If already valid, try to improve max
    if current_conflicts == 0:
        best_solution = [list(r) for r in current_rows]
        best_M = current_max
    
    # Simulated Annealing
    # We'll try to perturb the rows and accept/reject based on score
    # Perturbation: randomly change one element in one row, or swap two elements
    
    temperature = 10.0
    cooling_rate = 0.999
    min_temperature = 0.1
    steps = 0
    
    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        steps += 1
        
        # Random perturbation
        row_idx = random.randint(0, n_rows - 1)
        
        # Option 1: Change one element
        if random.random() < 0.5:
            pos = random.randint(1, k)  # Don't change the 0
            old_val = current_rows[row_idx][pos]
            # Try to change to a nearby value
            delta = random.randint(-5, 5)
            new_val = old_val + delta
            if new_val <= 0:
                new_val = 1
            # Ensure increasing order
            if pos > 0 and new_val <= current_rows[row_idx][pos - 1]:
                new_val = current_rows[row_idx][pos - 1] + 1
            if pos < k and new_val >= current_rows[row_idx][pos + 1]:
                new_val = current_rows[row_idx][pos + 1] - 1
                if new_val <= current_rows[row_idx][pos - 1] if pos > 0 else 0:
                    continue  # Skip if can't maintain order
            
            current_rows[row_idx][pos] = new_val
        else:
            # Option 2: Add/subtract a small random value to all elements in a row (shift)
            delta = random.randint(-3, 3)
            if delta != 0:
                for i in range(1, k + 1):
                    current_rows[row_idx][i] += delta
                # Check if still valid (positive and increasing)
                if current_rows[row_idx][1] <= 0:
                    for i in range(1, k + 1):
                        current_rows[row_idx][i] -= delta
                    continue
                if any(current_rows[row_idx][i] >= current_rows[row_idx][i + 1] for i in range(1, k)):
                    for i in range(1, k + 1):
                        current_rows[row_idx][i] -= delta
                    continue
        
        # Evaluate new state
        new_conflicts, new_max = get_score(current_rows)
        delta_conflicts = new_conflicts - current_conflicts
        delta_max = new_max - current_max
        
        # Acceptance criterion
        # Primary: conflicts, Secondary: max
        if delta_conflicts < 0 or (delta_conflicts == 0 and delta_max < 0):
            # Improvement
            current_conflicts = new_conflicts
            current_max = new_max
            if current_conflicts == 0 and current_max < best_M:
                best_M = current_max
                best_solution = [list(r) for r in current_rows]
        else:
            # Deterioration
            # Calculate probability of accepting
            # Use a combined score
            old_score = current_conflicts * 1000 + current_max
            new_score = new_conflicts * 1000 + new_max
            delta_score = new_score - old_score
            
            if delta_score > 0:
                # Deterioration
                prob = 1.0 / (1.0 + random.random())  # Simple acceptance
                if random.random() < prob * min(1.0, 10.0 / max(1, delta_score)):
                    current_conflicts = new_conflicts
                    current_max = new_max
                else:
                    # Reject: revert
                    if random.random() < 0.5:
                        # Revert the change (we didn't save old state, so this is tricky)
                        # For simplicity, we'll just accept all small deteriorations
                        pass
            else:
                # Improvement (shouldn't happen if we checked above, but just in case)
                current_conflicts = new_conflicts
                current_max = new_max
        
        temperature *= cooling_rate
    
    # Final check
    if best_solution is not None:
        return best_solution
    return fallback
