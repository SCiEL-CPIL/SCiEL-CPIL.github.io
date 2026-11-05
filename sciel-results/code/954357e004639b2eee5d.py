import time
import random
import math

def construct(n_rows, k):
    # n_rows is expected to be 4, k is expected to be 7 (length 8)
    # But we should handle general inputs if possible, though the task specifies (4,7).
    # The prompt says "The evaluator passes this task's n_rows and k."
    # We assume n_rows=4, k=7.
    
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # 1. Greedy Initialization
    def greedy_init():
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
        return rows

    best_rows = greedy_init()
    
    def get_cost(rows):
        """
        Returns (max_val, conflict_count)
        """
        diffs = []
        max_val = 0
        for row in rows:
            max_val = max(max_val, row[-1])
            for i in range(len(row)):
                ri = row[i]
                for j in range(i+1, len(row)):
                    diffs.append(row[j] - ri)
        
        # Check for duplicates
        # If there are duplicates, the set size will be smaller than the list size
        unique_diffs = set(diffs)
        conflicts = len(diffs) - len(unique_diffs)
        return max_val, conflicts

    best_cost = get_cost(best_rows)
    best_valid_cost = best_cost[0] if best_cost[1] == 0 else float('inf')
    
    # If initial is invalid (greedy should be valid, but just in case)
    if best_cost[1] > 0:
         # Force validity? Greedy guarantees validity.
         pass

    current_rows = [r[:] for r in best_rows]
    current_cost = best_cost
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.9999
    min_T = 0.1
    
    # Move: Shift a suffix of a row
    def propose_move(rows):
        # Pick a row
        r_idx = random.randint(0, n_rows - 1)
        # Pick a split point j (1 to k-1)
        # We shift indices j to k
        # j can be 1 to k-1
        # If j=1, we shift all except 0.
        # If j=k, we shift only the last element.
        j = random.randint(1, k)
        
        # Pick delta
        # Larger deltas are useful for escaping local minima
        # Use a heavy-tailed distribution
        delta = random.choice([-3, -2, -1, 1, 2, 3])
        
        # Check bounds
        row = rows[r_idx]
        # New value of row[j] would be row[j] + delta
        # We need row[j-1] < row[j] + delta
        # And row[j] + delta < row[j+1] (if j < k)
        # Wait, if we shift j..k, the relative order inside j..k is preserved.
        # So we only need to check the boundary at j-1 and j.
        # row[j-1] < row[j] + delta
        # And if j < k, row[j] + delta < row[j+1] is NOT required?
        # No, row[j+1] is also shifted by delta. So row[j]+delta < row[j+1]+delta is equivalent to row[j] < row[j+1], which is true.
        # So we only need row[j-1] < row[j] + delta.
        
        if row[j-1] < row[j] + delta:
            # Also ensure we don't create negative numbers? 
            # The problem says "increasing integers starting at 0".
            # It doesn't explicitly say non-negative, but "starting at 0" and "increasing" usually implies $0 < a_1 < ...$.
            # If row[j]+delta < 0, it's invalid for this task context (integers starting at 0 usually means positive integers after 0).
            # Let's assume positive integers.
            if row[j] + delta > 0:
                return r_idx, j, delta
        return None

    # Main SA Loop
    while time.time() < deadline:
        if T < min_T:
            # Reheat slightly or restart?
            # Let's just break and try to polish or restart from best
            # For now, break to avoid wasting time on tiny moves
            # Or restart with T=5.0
            T = 5.0 
            # Optionally reset to best
            # current_rows = [r[:] for r in best_rows]
            # current_cost = get_cost(current_rows)
            continue
            
        move = propose_move(current_rows)
        if move is None:
            T *= cooling_rate
            continue
            
        r_idx, j, delta = move
        
        # Apply move
        old_vals = [current_rows[r_idx][idx] for idx in range(j, k+1)]
        for idx in range(j, k+1):
            current_rows[r_idx][idx] += delta
        
        # Calculate new cost
        new_cost = get_cost(current_rows)
        
        # Delta Cost
        # We want to minimize max_val + penalty * conflicts
        # Let's define a scalar cost
        # If valid, cost = max_val
        # If invalid, cost = max_val + 1000 * conflicts
        
        def scalar_cost(c):
            return c[0] + 1000 * c[1]
            
        delta_c = scalar_cost(new_cost) - scalar_cost(current_cost)
        
        # Acceptance
        if delta_c < 0 or random.random() < math.exp(-delta_c / T):
            current_cost = new_cost
            # Update best if valid
            if new_cost[1] == 0:
                if new_cost[0] < best_valid_cost:
                    best_valid_cost = new_cost[0]
                    best_rows = [r[:] for r in current_rows]
        else:
            # Revert
            for idx in range(j, k+1):
                current_rows[r_idx][idx] = old_vals[idx - j]
                
        T *= cooling_rate

    return best_rows
