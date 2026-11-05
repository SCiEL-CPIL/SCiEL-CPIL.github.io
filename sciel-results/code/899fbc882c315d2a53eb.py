"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys

# EVOLVE-BLOCK-START
import random
import time
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within each row are distinct across all rows.
    Minimizes the largest entry (scope).
    
    Strategy: 
    1. Generate a valid initial solution using a simple greedy algorithm.
    2. Use Simulated Annealing to minimize the scope while maintaining validity (0 conflicts).
    """
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer
    
    # Helper to evaluate a configuration
    def evaluate(rows):
        seen = set()
        conflicts = 0
        max_scope = 0
        for row in rows:
            # Update max scope
            if row[-1] > max_scope:
                max_scope = row[-1]
            # Check differences
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        conflicts += 1
                    else:
                        seen.add(d)
        return conflicts, max_scope

    # 1. Generate a valid initial solution using greedy
    used = set()
    initial_rows = []
    for r in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
                x += 1
            else:
                x += 1
        initial_rows.append(row)
    
    # 2. Simulated Annealing
    current_rows = [row[:] for row in initial_rows]
    current_conflicts, current_scope = evaluate(current_rows)
    
    best_rows = [row[:] for row in current_rows]
    best_conflicts, best_scope = current_conflicts, current_scope
    
    T = 100.0
    T_min = 0.1
    cooling = 0.999
    
    # Number of iterations
    # Adjust based on problem size? For now, fixed high number, limited by time.
    num_iterations = 500000
    
    for i in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random move
        r = random.randint(0, n_rows - 1)
        j = random.randint(1, k)  # indices 1..k, index 0 is fixed at 0
        
        prev_val = current_rows[r][j - 1]
        
        if j < k:
            next_val = current_rows[r][j + 1]
            low = prev_val + 1
            high = next_val - 1
        else:
            low = prev_val + 1
            # Allow some expansion, but mostly try to shrink
            high = current_scope + 10
        
        if high < low:
            continue
        
        # Randomly pick a new value
        # Bias towards smaller values to help minimize scope?
        # Let's pick uniformly, but if the range is large, we might miss small values.
        # To encourage shrinking, we can sometimes pick from a smaller range.
        if random.random() < 0.5 and high > low + 10:
            # Try to pick a value closer to the lower bound
            high = min(high, low + 20)
        
        new_val = random.randint(low, high)
        
        if new_val == current_rows[r][j]:
            continue
        
        # Apply move
        old_val = current_rows[r][j]
        current_rows[r][j] = new_val
        
        # Evaluate
        new_conflicts, new_scope = evaluate(current_rows)
        
        # Cost function
        # Strongly prefer no conflicts
        old_cost = current_conflicts * 1000 + current_scope
        new_cost = new_conflicts * 1000 + new_scope
        
        delta_cost = new_cost - old_cost
        
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            # Accept with probability exp(-delta / T)
            if delta_cost > 1000:
                prob = 0
            else:
                prob = math.exp(-delta_cost / T)
            accept = random.random() < prob
        
        if accept:
            current_conflicts = new_conflicts
            current_scope = new_scope
            if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and current_scope < best_scope):
                best_conflicts = current_conflicts
                best_scope = current_scope
                best_rows = [row[:] for row in current_rows]
        else:
            # Revert
            current_rows[r][j] = old_val
            
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
    # If best solution has conflicts, fall back to initial valid one
    if best_conflicts > 0:
        return initial_rows
    else:
        return best_rows
# EVOLVE-BLOCK-END
