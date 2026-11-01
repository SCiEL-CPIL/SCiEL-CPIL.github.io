import random
import time

def construct(n_rows, k):
    row_len = k + 1
    max_time = 110
    start_time = time.time()
    
    best_rows = None
    best_cost = float('inf')
    
    # Increase restarts to explore more
    n_restarts = 100 
    
    for restart in range(n_restarts):
        if time.time() - start_time > max_time - 1:
            break
            
        # Use a different random seed for each restart to ensure diversity
        # We can just rely on the global random state being different, 
        # but explicitly seeding helps if we want reproducibility. 
        # Here we just call random functions.
        
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        success = True
        
        # We process steps one by one
        # To better "balance", we can sort rows by their current max value before each step?
        # Or just pick the min each time.
        
        for _ in range(n_rows * k):
            # Select row to extend
            # Find the row with the minimum max value among those not yet full
            min_max_val = float('inf')
            candidate_indices = []
            for i, r in enumerate(rows):
                if len(r) < row_len:
                    m = r[-1] # Since strictly increasing, max is last element
                    if m < min_max_val:
                        min_max_val = m
                        candidate_indices = [i]
                    elif m == min_max_val:
                        candidate_indices.append(i)
            
            if not candidate_indices:
                success = False
                break
                
            row_idx = random.choice(candidate_indices)
            row = rows[row_idx]
            
            # Find valid next values
            # We search for the smallest valid x, but keep a few options.
            # To align with "largest remaining differences", we might want to 
            # ensure we don't leave "holes" in the difference space that are too small?
            # Actually, "places the largest remaining differences first" is ambiguous.
            # Let's stick to the balancing heuristic.
            
            x = row[-1] + 1
            valid_xs = []
            
            # Limit search to avoid hanging
            # A good upper bound for the next element in a tight packing is not known a priori,
            # but we can search up to min_max_val + 100 or similar.
            limit = min_max_val + 100 + k*5
            
            while x <= limit and len(valid_xs) < 5:
                # Check if x is valid
                # New diffs: x - a for all a in row
                # Since row is sorted, x - a is decreasing as a increases.
                # We just need to check intersection with used_diffs.
                is_valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        is_valid = False
                        break
                
                if is_valid:
                    valid_xs.append(x)
                
                x += 1
            
            if not valid_xs:
                # If no valid x found in the window, try a larger window or fail
                # Expanding the window significantly increases time.
                # For now, fail this restart.
                success = False
                break
                
            # Pick a value
            # To introduce diversity, pick randomly from the first few.
            # If we always pick the smallest, we might get stuck in a local structure.
            chosen_x = random.choice(valid_xs)
            
            # Update
            for a in row:
                used_diffs.add(chosen_x - a)
            row.append(chosen_x)
            
        if success:
            current_cost = max(r[-1] for r in rows)
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [r[:] for r in rows]
                
    return best_rows
