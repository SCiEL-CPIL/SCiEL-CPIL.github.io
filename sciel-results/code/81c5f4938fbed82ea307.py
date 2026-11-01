import time
import random
import sys

# Set recursion limit high just in case, though we use iterative approaches
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows and k differences per row.
    Each row has k+1 elements starting at 0.
    Minimizes the largest entry (scope).
    """
    deadline = time.time() + 110  # Safety margin before 150s limit
    
    # Initial deterministic greedy construction to have a baseline
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    # Randomized Greedy + Local Search loop
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        # Randomize the search
        candidate = _randomized_greedy(n_rows, k, deadline)
        if candidate:
            # Local search to minimize scope
            improved = _local_search_minimize_scope(candidate, n_rows, k, deadline)
            if improved:
                curr_score = max(max(r) for r in improved)
                if curr_score < best_score:
                    best_score = curr_score
                    best_solution = improved
                    # If we hit the theoretical lower bound or a very good score, we might stop early
                    # The counting bound is 105. Best known is 112.
                    if best_score <= 105:
                        break
    
    return best_solution

def _greedy_construct(n_rows, k):
    """
    Deterministic greedy: always pick the smallest valid next mark.
    """
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Search up to a reasonable limit
            for x in range(current_max + 1, current_max + 2000):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _randomized_greedy(n_rows, k, deadline):
    """
    Randomized greedy: pick from a set of valid candidates with bias towards smaller values.
    """
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            if time.time() > deadline:
                return None
            
            # Find valid candidates in a window
            # Start with a small window, expand if necessary
            window_size = 50
            valid_candidates = []
            start_x = current_max + 1
            end_x = start_x + window_size
            
            for x in range(start_x, end_x):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= 10:
                        break
            
            if not valid_candidates:
                # If none found in small window, expand search (greedy fallback)
                for x in range(end_x, current_max + 2000):
                    new_diffs = []
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        valid_candidates.append((x, new_diffs))
                        break
            
            if not valid_candidates:
                return None
            
            # Pick randomly from valid candidates, biased towards smaller values
            # Weight is inversely proportional to value
            total_weight = sum(1.0/(x+1) for x, _ in valid_candidates)
            r = random.random() * total_weight
            cum = 0
            chosen_idx = 0
            for i, (x, nd) in enumerate(valid_candidates):
                cum += 1.0/(x+1)
                if r <= cum:
                    chosen_idx = i
                    break
            
            x, new_diffs = valid_candidates[chosen_idx]
            row.append(x)
            used_diffs.update(new_diffs)
            current_max = x
            
        rows.append(row)
    return rows

def _local_search_minimize_scope(rows, n_rows, k, deadline):
    """
    Local search to minimize the maximum entry (scope).
    Iteratively tries to replace elements with smaller valid values.
    """
    current_rows = [r[:] for r in rows]
    
    # Pre-compute all differences
    all_diffs = set()
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.add(r[j] - r[i])
                
    improved = False
    # Repeat until no improvement or time out
    while time.time() < deadline - 5:
        progress = False
        # Shuffle rows to avoid bias
        random.shuffle(current_rows)
        
        for r_idx in range(n_rows):
            if time.time() > deadline - 5:
                break
            row = current_rows[r_idx]
            
            # Identify the current global max
            global_max = max(max(r) for r in current_rows)
            
            # Try to minimize each element from largest to smallest
            # We focus on elements that contribute to the global max or are large
            for pos in range(len(row)-1, 0, -1): 
                old_val = row[pos]
                
                # If this element is not the global max, we might still reduce it
                # but reducing the global max is priority.
                # Let's try to find any smaller valid value.
                
                # Remove diffs involving old_val from the global set temporarily
                diffs_to_remove = set()
                for i in range(pos):
                    diffs_to_remove.add(old_val - row[i])
                for j in range(pos+1, len(row)):
                    diffs_to_remove.add(row[j] - old_val)
                
                # The set of diffs that must be avoided is all_diffs minus diffs_to_remove
                # But we also need to ensure new diffs don't collide with each other
                # and are not in the "other" diffs.
                
                # Efficient check:
                # 1. New value x must be > row[pos-1]
                # 2. New value x must be < row[pos+1] if pos < len(row)-1
                # 3. Diffs x - row[i] for i < pos must not be in (all_diffs - diffs_to_remove)
                # 4. Diffs row[j] - x for j > pos must not be in (all_diffs - diffs_to_remove)
                # 5. New diffs must be distinct among themselves (automatically true if row is strictly increasing and we check against existing set? No, we must check new diffs against each other too, but since they are generated from a single x and fixed other elements, they are distinct if the other elements are distinct. Wait, x-a1 and x-a2 are distinct if a1 != a2. a1 and a2 are distinct. So new diffs are distinct. We only need to check against the global set minus removed ones.)
                
                lower_bound = row[pos-1] + 1
                upper_bound = row[pos+1] - 1 if pos < len(row) - 1 else old_val - 1
                
                # We want to find the smallest x in [lower_bound, upper_bound] that is valid.
                # Actually, we just want ANY x < old_val that is valid to reduce scope.
                # To minimize scope, we should try to find the smallest possible x.
                
                found_new_val = -1
                # Search from lower_bound up to old_val - 1
                # To speed up, we can check in random order or sequential.
                # Sequential is better for finding the *minimum* possible value.
                for new_val in range(lower_bound, old_val):
                    # Check validity
                    valid = True
                    new_diffs = set()
                    
                    # Check diffs with previous elements
                    for i in range(pos):
                        d = new_val - row[i]
                        if d in all_diffs and d not in diffs_to_remove:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                        
                    # Check diffs with next elements
                    for j in range(pos+1, len(row)):
                        d = row[j] - new_val
                        if d in all_diffs and d not in diffs_to_remove:
                            valid = False
                            break
                        new_diffs.add(d)
                        
                    if valid:
                        found_new_val = new_val
                        break # We found the smallest valid value, which is best for minimization
                
                if found_new_val != -1:
                    # Update the row and the global diff set
                    # Remove old diffs
                    all_diffs.difference_update(diffs_to_remove)
                    
                    # Add new diffs
                    for i in range(pos):
                        all_diffs.add(found_new_val - row[i])
                    for j in range(pos+1, len(row)):
                        all_diffs.add(row[j] - found_new_val)
                    
                    row[pos] = found_new_val
                    progress = True
                    improved = True
                    # If we reduced the global max, we might want to restart the scan
                    # But continuing is fine, it will catch other reductions.
        
        if not progress:
            # No further improvements found in this pass
            break
            
    return current_rows if improved else None

# Keep the original entry point name and signature
# The function construct is already defined above.

if __name__ == "__main__":
    # Quick self-test
    start = time.time()
    res = construct(7, 5)
    end = time.time()
    if res:
        print(f"Result: {res}")
        print(f"Max entry: {max(max(r) for r in res)}")
        print(f"Time taken: {end - start:.2f}s")
        # Validate
        used = set()
        valid = True
        for r in res:
            if r[0] != 0:
                valid = False
                break
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        valid = False
                        break
                    used.add(d)
            if not valid:
                break
        print(f"Valid: {valid}")
    else:
        print("No solution found")
