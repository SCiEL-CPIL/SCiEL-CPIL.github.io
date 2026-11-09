import sys
import time
import random
import copy

def construct(n_rows, k):
    # Parameters
    # n_rows = 4, k = 7 (from task description, though function takes args)
    # Each row has k+1 elements.
    
    start_time = time.time()
    timeout = 110.0
    
    best_max = float('inf')
    best_solution = None
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_max(rows):
        return max(max(r) for r in rows)

    def greedy_randomized(max_limit=None):
        # Generate a valid solution using greedy with some randomization
        rows = []
        used = set()
        for r_idx in range(n_rows):
            row = [0]
            last = 0
            for _ in range(k):
                found = False
                # Try values starting from last+1
                # To add randomness, we can skip some small values sometimes?
                # Or just pick the first valid one, but shuffle the order of checks?
                # No, we need strict increase.
                # Let's try a range of candidates.
                candidates = []
                for x in range(last + 1, max_limit if max_limit else 200):
                    # Check conflicts
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    if len(candidates) > 20:
                        break
                
                if not candidates:
                    # No candidate found in this range, try larger
                    # For simplicity, just extend range
                    for x in range(last + 1, 500):
                        valid = True
                        for a in row:
                            d = x - a
                            if d in used:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                        if len(candidates) > 20:
                            break
                            
                if not candidates:
                    return None # Failed to construct
                
                # Pick a random candidate from the first few to introduce diversity
                # Picking the smallest is deterministic. Picking a random one from top 5 helps.
                pick_idx = random.randint(0, min(4, len(candidates)-1))
                x = candidates[pick_idx]
                
                # Update used
                for a in row:
                    used.add(x - a)
                
                row.append(x)
                last = x
            rows.append(row)
        return rows

    def local_search(rows, time_limit):
        # Hill climbing to minimize max element
        current_rows = [r[:] for r in rows]
        current_max = get_max(current_rows)
        no_improve_count = 0
        limit = 500 # Max moves before restarting local search phase
        
        while time.time() - start_time < time_limit and no_improve_count < limit:
            # Pick a random row and index
            r = random.randint(0, n_rows - 1)
            i = random.randint(1, k) # indices 1..k
            
            old_val = current_rows[r][i]
            
            # Determine range for new value
            min_val = current_rows[r][i-1] + 1
            max_val = current_rows[r][i+1] - 1 if i < k else current_max
            
            # We want to decrease the value to reduce max, or just move to explore
            # If i == k (last element), we strongly prefer decreasing.
            if i == k:
                # Try to decrease
                for _ in range(10):
                    new_val = random.randint(min_val, max_val)
                    if new_val >= old_val:
                        continue
                    # Check validity
                    candidate_rows = [row[:] for row in current_rows]
                    candidate_rows[r][i] = new_val
                    if is_valid(candidate_rows):
                        new_max = get_max(candidate_rows)
                        if new_max < current_max:
                            current_rows = candidate_rows
                            current_max = new_max
                            no_improve_count = 0
                            break
                        elif new_max == current_max and random.random() < 0.1:
                            current_rows = candidate_rows
                            no_improve_count += 1
                            break
                    no_improve_count += 1
            else:
                # For internal elements, just random move within bounds
                if min_val < max_val:
                    new_val = random.randint(min_val, max_val)
                    if new_val != old_val:
                        candidate_rows = [row[:] for row in current_rows]
                        candidate_rows[r][i] = new_val
                        if is_valid(candidate_rows):
                            new_max = get_max(candidate_rows)
                            if new_max <= current_max:
                                current_rows = candidate_rows
                                if new_max < current_max:
                                    current_max = new_max
                                    no_improve_count = 0
                                else:
                                    no_improve_count += 1
                            else:
                                no_improve_count += 1
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                    
        return current_rows, current_max

    # Main loop
    while time.time() - start_time < timeout:
        # Generate a new random greedy solution
        sol = greedy_randomized()
        if sol is None:
            continue
        
        curr_max = get_max(sol)
        if curr_max < best_max:
            best_max = curr_max
            best_solution = [r[:] for r in sol]
        
        # Run local search to improve
        # Allocate remaining time / 10 to each local search attempt
        remaining = timeout - (time.time() - start_time)
        if remaining < 1:
            break
        ls_time = min(5.0, remaining / 2) # Spend up to 5s on local search
        
        improved_sol, improved_max = local_search(sol, start_time + ls_time)
        
        if improved_max < best_max:
            best_max = improved_max
            best_solution = [r[:] for r in improved_sol]
            
    return best_solution
