import time
import random
import sys

def greedy_construct(n_rows, k):
    """
    Standard greedy construction to establish a baseline.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            diffs = {x - a for a in row}
            if not diffs & used:
                row.append(x)
                used.update(diffs)
        rows.append(row)
    return rows

def calculate_cost(rows, n_rows, k):
    """
    Calculate a cost function for the current set of rows.
    We want to minimize:
    1. Number of duplicate differences (collisions)
    2. The maximum entry (scope)
    
    We combine these into a single scalar cost.
    Since validity is hard to achieve, we prioritize removing collisions.
    """
    all_diffs = []
    max_entry = 0
    
    for r in rows:
        if not r:
            continue
        max_entry = max(max_entry, max(r))
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.append(r[j] - r[i])
                
    # Count duplicates
    diff_counts = {}
    for d in all_diffs:
        diff_counts[d] = diff_counts.get(d, 0) + 1
        
    collisions = sum(1 for count in diff_counts.values() if count > 1)
    
    # Cost: Heavily penalize collisions, then penalize scope
    # We use a weighted sum. If collisions > 0, that dominates.
    # To guide search towards smaller scopes even with collisions, 
    # we can use: cost = collisions * 10000 + max_entry
    return collisions * 10000 + max_entry

def local_search(rows, n_rows, k, time_limit, seed=0):
    """
    Perform randomized local search (Simulated Annealing style) 
    to minimize collisions and scope.
    """
    random.seed(seed)
    start_time = time.time()
    
    # Initialize with a randomized valid structure? 
    # The input 'rows' should be a valid starting point (e.g., from greedy) 
    # or we can start from scratch. 
    # Let's assume we pass a valid or near-valid set. 
    # Actually, for robustness, let's start from the greedy result which is valid.
    
    current_rows = [list(r) for r in rows]
    current_cost = calculate_cost(current_rows, n_rows, k)
    
    best_rows = [list(r) for r in current_rows]
    best_cost = current_cost
    
    temperature = 1000.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor
        # Move types:
        # 1. Change a single element in a row (preserving order)
        # 2. Swap two elements in different rows? (Complex to maintain order)
        # 3. Insert/Remove? No, size is fixed.
        
        # Let's stick to changing one element in one row.
        row_idx = random.randint(0, n_rows - 1)
        # Don't change the first element (must be 0)
        if len(current_rows[row_idx]) <= 1:
            continue
        pos_idx = random.randint(1, len(current_rows[row_idx]) - 1)
        
        old_val = current_rows[row_idx][pos_idx]
        
        # Choose a new value
        # Heuristic: Try values close to current, or random within a range
        # Range: [prev_val + 1, next_val - 1] or [prev_val + 1, current_max + 10]
        prev_val = current_rows[row_idx][pos_idx - 1]
        next_val = current_rows[row_idx][pos_idx + 1] if pos_idx + 1 < len(current_rows[row_idx]) else float('inf')
        
        # We want to keep the row strictly increasing.
        min_val = prev_val + 1
        max_val = min(next_val - 1, max(max(r) for r in current_rows) + 20)
        
        if min_val > max_val:
            continue
            
        # Randomly pick a new value in range
        # Bias towards smaller values to reduce scope
        new_val = random.randint(min_val, max_val)
        
        if new_val == old_val:
            continue
            
        # Apply move
        current_rows[row_idx][pos_idx] = new_val
        
        # Calculate new cost
        new_cost = calculate_cost(current_rows, n_rows, k)
        
        delta = new_cost - current_cost
        
        # Metropolis criterion
        if delta < 0 or random.random() < pow(2.718281828, -delta / max(temperature, 0.1)):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [list(r) for r in current_rows]
                # If we reached a valid state (cost < 10000 implies 0 collisions? 
                # No, cost = collisions*10000 + max. 
                # If collisions=0, cost = max. So if best_cost < 10000, it's valid.)
                # We can't stop immediately because we want to minimize scope further.
        else:
            # Revert
            current_rows[row_idx][pos_idx] = old_val
            
        temperature *= cooling_rate
        if temperature < 1.0:
            temperature = 1.0
            
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # 1. Greedy Baseline
    best_rows = greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in best_rows)
    
    # Verify baseline
    all_diffs = set()
    valid = True
    for r in best_rows:
        if len(r) != k + 1:
            valid = False
            break
        if r[0] != 0:
            valid = False
            break
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    valid = False
                    break
                all_diffs.add(d)
        if not valid:
            break
    
    if not valid:
        # Should not happen with greedy, but just in case
        return best_rows

    # 2. Local Search Improvement
    # We run multiple independent local search instances with different seeds
    # to explore different basins of attraction.
    
    # Time budget per instance
    # We have ~130s. Let's do 5 instances of 25s each? 
    # Or one long run? 
    # Parallelism is not available, so sequential.
    # Let's do a few restarts.
    
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts - 2 # 2s safety
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        seed = i * 1000 + 42
        # Pass a copy of the current best or the greedy baseline?
        # Starting from greedy baseline ensures we don't lose the valid state.
        # But local search starts from a valid state and tries to improve scope.
        # If it gets stuck in a local minimum with higher scope but valid, 
        # it will return that. We keep the best found.
        
        candidate_rows = local_search(best_rows, n_rows, k, time_per_restart, seed)
        
        # Validate candidate
        all_diffs_c = set()
        valid_c = True
        max_c = 0
        for r in candidate_rows:
            if len(r) != k + 1:
                valid_c = False
                break
            if r[0] != 0:
                valid_c = False
                break
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in all_diffs_c:
                        valid_c = False
                        break
                    all_diffs_c.add(d)
                    if r[j] > max_c:
                        max_c = r[j]
            if not valid_c:
                break
        
        if valid_c:
            if max_c < best_scope:
                best_scope = max_c
                best_rows = candidate_rows
            # If we found a valid solution, we can try to improve from this new best
            # in the next restart? 
            # local_search takes 'rows' as input. If we pass best_rows, 
            # it starts from the current best. This allows incremental improvement.
            # However, if we get stuck, restarting from the original greedy 
            # might find a different path. 
            # Let's alternate: sometimes start from best, sometimes from greedy.
            # For simplicity, let's always start from the current best_rows 
            # to exploit the "warm start" if it's good, but the randomness 
            # in local_search should handle exploration.
            # Actually, passing best_rows means the "current" state in LS 
            # is the best found so far. This is good for refinement.
            
    return best_rows

if __name__ == "__main__":
    # Simple test
    rows = construct(4, 7)
    print(rows)
