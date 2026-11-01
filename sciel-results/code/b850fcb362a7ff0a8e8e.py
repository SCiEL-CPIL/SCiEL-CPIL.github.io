import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    # Lower bound for M: n_rows * k * (k+1) / 2
    # For 7, 5: 7 * 15 = 105
    min_M = n_rows * k * (k + 1) // 2
    
    # Upper bound heuristic: usually slightly above lower bound
    # Shehadeh et al. found 112 for (7,5).
    # Let's start searching from min_M upwards.
    
    best_solution = None
    current_M = min_M
    
    # Max M to try before giving up on optimization and returning a valid (possibly larger) solution
    # We can estimate a safe upper bound. 
    # A simple greedy construction always finds a solution if M is large enough.
    # O(k^2 * n_rows) is a rough heuristic for the range needed if we don't optimize perfectly.
    max_M_to_try = min_M + 50 
    
    while current_M <= max_M_to_try and time.time() < deadline - 10.0:
        
        # Try to find a valid packing of n_rows Sidon sets of size k+1 in [0, current_M]
        # We use a randomized greedy approach with backtracking for the last few rows.
        
        success = False
        # Try multiple random restarts for this M
        num_restarts = 50 if current_M == min_M else 20
        if time.time() > deadline - 15.0:
            num_restarts = 1
            
        for _ in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            used_diffs = set()
            rows = []
            
            # Shuffle the order in which we pick rows? No, rows are indistinguishable.
            # We just need to pick n_rows disjoint sets.
            
            # To improve success rate, we can try to pick the "tightest" sets first?
            # Or just random greedy.
            
            # Heuristic: Pick elements for each row one by one.
            # For better performance, we can pre-compute all valid Sidon sets of size k+1 in [0, current_M]?
            # That might be too slow if current_M is large.
            # Instead, we generate a row on the fly.
            
            valid = True
            for r in range(n_rows):
                row = [0]
                # Generate the rest of the row
                # We need k more elements.
                # We use a recursive helper with backtracking for the current row only.
                
                def gen_row(current_row, used_diffs_local):
                    if time.time() > deadline - 5.0:
                        return None
                    if len(current_row) == k + 1:
                        return list(current_row)
                    
                    last = current_row[-1]
                    # Try next elements
                    # We can randomize the order of candidates to escape local traps
                    # Candidates must be > last and <= current_M
                    # And their diffs to all existing elements in current_row must be unused in used_diffs_local (which includes global used_diffs + local diffs)
                    
                    # Optimization: Pre-compute valid next steps?
                    # Let's just iterate from last+1 to current_M.
                    # To add randomness, we can shuffle the range?
                    # Shuffling 100+ items is cheap.
                    
                    # However, iterating sequentially is faster for the first few.
                    # Let's try a mix: sequential with occasional jumps?
                    # Or just sequential, relying on restarts.
                    
                    # Actually, randomization in the choice of the *next* element is key.
                    # Let's collect valid candidates and pick one.
                    candidates = []
                    for x in range(last + 1, current_M + 1):
                        new_diffs = []
                        valid_candidate = True
                        for y in current_row:
                            d = x - y
                            if d in used_diffs_local:
                                valid_candidate = False
                                break
                            # Check if d is already in new_diffs (internal collision)
                            if d in new_diffs:
                                valid_candidate = False
                                break
                            new_diffs.append(d)
                        
                        if valid_candidate:
                            candidates.append((x, new_diffs))
                    
                    if not candidates:
                        return None
                    
                    # Pick a candidate
                    # To encourage finding a solution, maybe pick the smallest?
                    # Or random? Random helps with global packing.
                    # Let's pick randomly from the top few smallest to balance tightness and diversity.
                    # Or just random from all valid.
                    # Random from all valid is safer for packing.
                    idx = random.randint(0, len(candidates) - 1)
                    x, new_diffs = candidates[idx]
                    
                    # Update local state
                    new_used = used_diffs_local.union(new_diffs)
                    res = gen_row(current_row + [x], new_used)
                    if res:
                        return res
                    return None

                # Call gen_row
                # Note: used_diffs is global. We need to pass a copy or handle backtracking properly.
                # Since gen_row is recursive and returns None on failure, we can't easily "undo" the global used_diffs if we don't use a persistent state.
                # But gen_row above takes used_diffs_local.
                # We need to start with the global used_diffs for this row.
                
                row_res = gen_row([0], set(used_diffs))
                
                if row_res is None:
                    valid = False
                    break
                
                rows.append(row_res)
                # Update global used_diffs
                for i in range(len(row_res)):
                    for j in range(i + 1, len(row_res)):
                        used_diffs.add(row_res[j] - row_res[i])
            
            if valid:
                success = True
                # We found a solution for this M
                # Check if it's better than best_solution
                if best_solution is None or max(max(r) for r in rows) < max(max(r) for r in best_solution):
                    best_solution = rows
                # Since we are increasing M, the first solution found for a given M is valid.
                # We can return it immediately if we want to save time, 
                # but maybe a later restart finds a smaller max? No, max is bounded by current_M.
                # So any solution has max <= current_M.
                # If we found one, we can return it immediately as it's optimal for this M.
                # And since we start from min_M, the first M that works is the best we can find with this method.
                return rows
        
        # If no solution found for this M, increase M
        current_M += 1

    # If we didn't find a solution in the loop (e.g. timeout or M exceeded), 
    # we must return a valid solution.
    # Fallback: Greedy construction with large M.
    if best_solution is not None:
        return best_solution
    
    # Construct a valid solution with a large M
    # M = 2 * min_M is usually safe.
    safe_M = min_M + 100
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            valid = True
            new_diffs = []
            for y in row:
                d = x - y
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                row.append(x)
                used_diffs.update(new_diffs)
            x += 1
        rows.append(row)
        
    return rows
