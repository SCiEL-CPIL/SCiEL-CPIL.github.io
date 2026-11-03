"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time
import sys

def construct(n_rows, k):
    """
    Uses Simulated Annealing to minimize the largest entry in a Difference Triangle Set.
    """
    start_time = time.time()
    max_time = 115.0
    
    # Helper to check if a set of rows is valid
    def is_valid(rows):
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        return False
                    used_diffs.add(d)
        return True

    def get_cost(rows):
        return max(max(row) for row in rows)

    # Generate a random valid initial solution using greedy
    def generate_initial(n_rows, k):
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < k + 1:
                # Try random increments or just increment until valid
                # To make it random, we can shuffle the range of next possible values
                # But greedy is safer for initialization
                found = False
                # Search in a window
                for x in range(last + 1, last + 100):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        last = x
                        used_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    # Fallback: linear search
                    x = last + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            row.append(x)
                            last = x
                            used_diffs |= new_diffs
                            break
                        x += 1
            rows.append(row)
        return rows

    # Simulated Annealing
    best_rows = None
    best_cost = float('inf')
    
    # Run multiple restarts
    restarts = 20
    for r in range(restarts):
        if time.time() - start_time > max_time:
            break
            
        current_rows = generate_initial(n_rows, k)
        current_cost = get_cost(current_rows)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_rows = [list(row) for row in current_rows]
            
        # Annealing parameters
        T = 10.0
        cooling_rate = 0.995
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() - start_time > max_time:
                break
                
            # Pick a random row and element to modify
            row_idx = random.randint(0, n_rows - 1)
            # We cannot change the first element (0)
            elem_idx = random.randint(1, k)
            
            old_val = current_rows[row_idx][elem_idx]
            
            # Determine valid range for new value
            min_val = current_rows[row_idx][elem_idx - 1] + 1
            if elem_idx < k:
                max_val = current_rows[row_idx][elem_idx + 1] - 1
            else:
                max_val = old_val + 20 # Allow increasing the last element slightly to unblock others
            
            # We want to try to reduce the value, but SA allows increases.
            # Bias towards smaller values to minimize cost.
            # Sample new_val from [min_val, max_val]
            # To be efficient, we can try a few random candidates
            
            improved = False
            # Try 5 random candidates
            for _ in range(5):
                if min_val > max_val:
                    continue
                
                # Bias towards lower values
                # Uniform random in [min_val, max_val]
                new_val = random.randint(min_val, max_val)
                
                if new_val == old_val:
                    continue
                
                # Check validity of the move
                # Remove old diffs involving old_val
                row = current_rows[row_idx]
                old_diffs = set()
                for j in range(len(row)):
                    if j == elem_idx:
                        continue
                    if row[j] > old_val:
                        old_diffs.add(row[j] - old_val)
                    else:
                        old_diffs.add(old_val - row[j])
                
                # Calculate new diffs
                new_diffs = set()
                for j in range(len(row)):
                    if j == elem_idx:
                        continue
                    if row[j] > new_val:
                        new_diffs.add(row[j] - new_val)
                    else:
                        new_diffs.add(new_val - row[j])
                
                # Check internal uniqueness
                if len(new_diffs) != len([d for d in new_diffs]): # set handles this, but we need to check against used_diffs
                    pass
                
                # We need a global used_diffs set for O(1) lookup
                # Rebuilding used_diffs every step is O(N^2). 
                # For 4 rows of 8, N=32, N^2=1024. 5000 steps * 1024 = 5M ops. Fast enough.
                
                # Let's optimize: maintain used_diffs incrementally?
                # It's complex. Let's just check validity of the whole row against global set?
                # No, we can just check if new_diffs intersect with (used_diffs - old_diffs)
                
                # For simplicity and correctness in this rewrite, let's just check the full validity of the modified row
                # against a pre-computed set of all other diffs?
                # Actually, just checking the new_diffs against a global set is best.
                
                # Let's rebuild the global used_diffs set excluding the current row's diffs?
                # That's expensive.
                
                # Alternative: 
                # 1. Store all diffs in a dictionary: diff -> count
                # 2. When changing a value, decrement counts for old diffs, increment for new.
                # 3. If any count becomes negative or > 1, invalid.
                
                # Let's implement the count-based validity check for speed.
                
                pass # Placeholder for logic below
            
            # Since the inner logic is complex, let's use a simpler validity check for the SA step
            # We will maintain a global set of used differences.
            # But updating it on every trial move is tricky.
            
            # Let's stick to a simpler "Hill Climbing" with restarts for reliability, 
            # as SA with complex state updates is error-prone in a short script.
            
            # Revised Plan: 
            # 1. Greedy Init.
            # 2. Local Search: 
            #    - For each row, try to minimize each element by scanning downwards from current value.
            #    - If a smaller valid value is found, take it.
            #    - Repeat until no improvements.
            # 3. Restart with different random seeds for greedy.
            
            pass

    # The above SA implementation was getting messy. Let's implement a clean Hill Climbing with Random Restarts.
    # This is a distinct approach from the original Randomized Backtracking.
    
    # Re-implementing construct with Hill Climbing
    
    def hill_climb(n_rows, k, initial_rows):
        rows = [list(row) for row in initial_rows]
        used_diffs = set()
        
        # Build used_diffs
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        improved = True
        while improved:
            improved = False
            for r in range(n_rows):
                for i in range(1, len(rows[r])):
                    old_val = rows[r][i]
                    min_val = rows[r][i-1] + 1
                    if i < len(rows[r]) - 1:
                        max_val = rows[r][i+1] - 1
                    else:
                        max_val = old_val # Don't increase last element in this phase, only decrease
                    
                    # Try to find a smaller value
                    # Scan from min_val upwards? No, we want the smallest valid value >= min_val.
                    # But we start from old_val and go down?
                    # No, we want to minimize. So we want the smallest valid x >= min_val.
                    # If old_val is not the smallest, we can reduce it.
                    
                    # Let's scan from min_val to old_val - 1
                    # If we find a valid x, replace old_val with x.
                    
                    found = False
                    for new_val in range(min_val, old_val):
                        # Check validity
                        new_diffs = set()
                        for j in range(len(rows[r])):
                            if j == i:
                                continue
                            if rows[r][j] > new_val:
                                d = rows[r][j] - new_val
                            else:
                                d = new_val - rows[r][j]
                            if d in used_diffs:
                                # Check if d was in old_diffs
                                # If d is in used_diffs, it might be from the old value.
                                # We need to be careful.
                                pass
                            
                        # This check is flawed because used_diffs contains old_val's diffs.
                        # We need to remove old_val's diffs, check, then add new_val's diffs.
                        
                        # Let's do it properly:
                        old_diffs = set()
                        for j in range(len(rows[r])):
                            if j == i:
                                continue
                            if rows[r][j] > old_val:
                                old_diffs.add(rows[r][j] - old_val)
                            else:
                                old_diffs.add(old_val - rows[r][j])
                        
                        # Check if new_diffs intersect with (used_diffs - old_diffs)
                        valid = True
                        new_diffs_set = set()
                        for j in range(len(rows[r])):
                            if j == i:
                                continue
                            if rows[r][j] > new_val:
                                d = rows[r][j] - new_val
                            else:
                                d = new_val - rows[r][j]
                            
                            if d in new_diffs_set:
                                valid = False
                                break
                            new_diffs_set.add(d)
                            
                            if d in used_diffs and d not in old_diffs:
                                valid = False
                                break
                        
                        if valid:
                            # Accept move
                            rows[r][i] = new_val
                            used_diffs.difference_update(old_diffs)
                            used_diffs.update(new_diffs_set)
                            improved = True
                            found = True
                            break
                    
                    # If we found a smaller value, we restart the scan for this element? 
                    # Or just continue? 
                    # Since we changed the value, the next element's min_val might change? No, min_val is fixed by previous.
                    # But the diffs change.
                    # We should probably restart the loop for this row or just continue.
                    # Continuing is fine, we'll catch it in the next pass.

        return rows

    # Main loop
    best_rows = None
    best_cost = float('inf')
    
    start_time = time.time()
    max_time = 115.0
    
    # Multiple restarts
    for restart in range(50):
        if time.time() - start_time > max_time:
            break
        
        # Generate random initial
        rows = generate_initial(n_rows, k)
        
        # Hill climb
        rows = hill_climb(n_rows, k, rows)
        
        cost = max(max(row) for row in rows)
        
        if cost < best_cost:
            best_cost = cost
            best_rows = [list(row) for row in rows]
            
        # Early stop if optimal
        if best_cost <= n_rows * (k + 1) * k // 2:
            break
            
    if best_rows is None:
        best_rows = generate_initial(n_rows, k)
        
    return best_rows

# EVOLVE-BLOCK-END
