"""
Strategy: Simulated Annealing with global difference-set representation.
We model the problem as minimizing the maximum scope (largest entry) while
ensuring all differences across all rows are distinct.
State: List of rows (each row is a list of k+1 integers starting at 0).
Cost: Scope + Penalty for invalidity (conflicting differences).
Moves: Move an element in a row to a new position, or swap elements between rows.
"""
import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Compute set of differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i+1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_diffs(rows):
    """Compute global set of differences and check for conflicts."""
    all_diffs = set()
    conflicts = 0
    for row in rows:
        r_diffs = _row_diffs(row)
        # Check internal conflicts (shouldn't happen if we maintain valid rows, 
        # but for SA we might allow invalid states temporarily? 
        # Actually, let's keep rows internally valid to simplify, 
        # or just count all conflicts as penalty.
        # Let's count total number of duplicate differences globally.
        # A simpler penalty: number of differences that appear more than once.
        # Or just: len(all_diffs) should be n_rows * k(k+1)/2.
        # Penalty = (n_rows * k(k+1)/2) - len(all_diffs)
        # But we also need to ensure rows are strictly increasing.
        
        # Let's use a simpler cost function for SA:
        # 1. Scope (max value in any row)
        # 2. Number of conflicting differences (duplicates)
        
        for d in r_diffs:
            if d in all_diffs:
                conflicts += 1 # Count each duplicate occurrence
            else:
                all_diffs.add(d)
    return scope, conflicts

def _scope(rows):
    return max(max(r) for r in rows)

def _is_valid(rows):
    """Check if the solution is valid: strictly increasing, distinct global diffs."""
    all_diffs = set()
    for row in rows:
        if not all(row[i] < row[i+1] for i in range(len(row)-1)):
            return False
        if row[0] != 0:
            return False
        r_diffs = _row_diffs(row)
        if len(r_diffs) != len(row)*(len(row)-1)//2:
            return False
        if all_diffs & r_diffs:
            return False
        all_diffs |= r_diffs
    return True

def _cost(rows):
    """
    Cost function for SA.
    We want to minimize scope.
    We heavily penalize invalidity (conflicts).
    """
    scope_val = _scope(rows)
    all_diffs = set()
    conflicts = 0
    for row in rows:
        # Check strictly increasing
        for i in range(len(row)-1):
            if row[i] >= row[i+1]:
                conflicts += 1000 # Heavy penalty for non-increasing
        if row[0] != 0:
            conflicts += 1000
        r_diffs = _row_diffs(row)
        # Check internal distinctness
        if len(r_diffs) != len(row)*(len(row)-1)//2:
            conflicts += 1000
        for d in r_diffs:
            if d in all_diffs:
                conflicts += 1
            else:
                all_diffs.add(d)
    
    # Cost = Scope + 10 * Conflicts
    # If valid, Cost = Scope
    # If invalid, Cost is large
    return scope_val + 10 * conflicts

def _greedy_construct(n_rows, k):
    """
    Fast greedy construction to get a valid starting point.
    """
    rows = []
    used_diffs = set()
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        # Try to build row with smallest possible values
        # We need k more elements.
        # Simple greedy: pick smallest x > last such that diffs are new
        while len(row) < k + 1:
            x = row[-1] + 1
            valid = False
            while not valid:
                # Check if adding x creates conflicts
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    # Also check internal distinctness of new_diffs
                    if len(new_diffs) == len(row):
                        valid = True
                x += 1
            row.append(x)
            used_diffs |= {x - a for a in row}
        rows.append(row)
    return rows

def _local_search_refine(rows, deadline):
    """
    Hill climbing to reduce scope.
    Try to move elements to smaller positions if valid.
    """
    current_cost = _cost(rows)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to reduce the maximum element in the row with the largest max
        max_row_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
        row = rows[max_row_idx]
        
        # Try to move the last element to a smaller value
        old_last = row[-1]
        # Try values from old_last - 1 down to row[-2] + 1
        for new_last in range(old_last - 1, row[-2], -1):
            new_row = row[:-1] + [new_last]
            # Check if this row is internally valid
            if not all(new_row[i] < new_row[i+1] for i in range(len(new_row)-1)):
                continue
            
            # Check global validity
            # Temporarily replace
            rows[max_row_idx] = new_row
            if _is_valid(rows):
                improved = True
                break
            else:
                rows[max_row_idx] = row # Restore
        
        # Also try moving other elements? 
        # For now, just the last element is a good first step.
        # If we want more, we can try swapping elements in different rows.
        
    return rows

def _simulated_annealing(rows, n_rows, k, deadline):
    """
    SA to minimize scope.
    """
    current_rows = [list(r) for r in rows]
    current_cost = _cost(current_rows)
    best_rows = [list(r) for r in rows]
    best_cost = current_cost
    
    start_temp = 50.0
    end_temp = 0.1
    # Number of steps we can afford
    # Estimate: 1 step ~ 1ms? Let's just run until deadline.
    
    t = start_temp
    # We need to decrease temp over time.
    # Let's do a fixed number of iterations if we can estimate time, 
    # or just decrement temp by a small amount each step.
    # Better: track elapsed time and set temp based on remaining time?
    # Simple: decrement t by 0.01 each step.
    
    while time.time() < deadline:
        # Generate a neighbor
        # Strategy: Pick a random row, pick a random element (not 0), 
        # and move it to a random new position (swap with another element or move to gap)
        # To keep rows sorted, we can:
        # 1. Remove element at index i (i>0)
        # 2. Insert it at index j (0 < j < len(row))
        # 3. Sort the row? No, that changes the set of differences? 
        #    Wait, the set of differences depends on the values, not the order in the list, 
        #    BUT the problem requires "increasing integers". 
        #    So the row is a set of values. The order is fixed by sorting.
        #    So "moving an element" means changing its value.
        
        # Let's try: Change value of a random element in a random row.
        # Pick row r
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Pick element index (1 to k)
        e_idx = random.randint(1, k)
        old_val = row[e_idx]
        
        # Try to change to a nearby value
        # Random walk
        step = random.randint(-10, 10)
        new_val = old_val + step
        
        # Constraints:
        # 1. new_val > row[e_idx-1]
        # 2. new_val < row[e_idx+1] (if e_idx < k)
        # 3. new_val > 0
        
        min_val = row[e_idx-1] + 1
        max_val = row[e_idx+1] - 1 if e_idx < k else 10000 # Large upper bound
        
        if new_val < min_val:
            new_val = min_val
        if new_val > max_val:
            new_val = max_val
            
        if new_val == old_val:
            continue
            
        # Apply change
        row[e_idx] = new_val
        
        new_cost = _cost(current_rows)
        
        # Acceptance criterion
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / t):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [list(r) for r in current_rows]
        else:
            # Reject
            row[e_idx] = old_val
            
        # Decrease temperature
        t *= 0.999
        if t < end_temp:
            t = end_temp
            
        # Occasionally do a local search
        if random.random() < 0.01:
             # Quick local search on current best? Or current state?
             # Let's just refine the best found so far
             pass
             
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 145.0 # Safety margin
    
    # 1. Greedy Initialization
    rows = _greedy_construct(n_rows, k)
    
    # 2. Local Search Refinement
    rows = _local_search_refine(rows, start_time + 10.0) # Spend 10s on LS
    
    # 3. Simulated Annealing
    if time.time() < DEADLINE - 5.0:
        rows = _simulated_annealing(rows, n_rows, k, DEADLINE - 5.0)
        
    # 4. Final Local Search
    rows = _local_search_refine(rows, DEADLINE)
    
    # Ensure validity
    if not _is_valid(rows):
        # Fallback to greedy if SA broke it (shouldn't happen if we restore on reject, 
        # but best_rows might be invalid if we saved an intermediate state? 
        # No, we only save best if cost is lower, and cost includes penalty. 
        # But _is_valid is stricter than cost? 
        # Cost = Scope + 10*Conflicts. 
        # If Conflicts=0, Cost=Scope. 
        # So if best_cost is minimal, it should be valid. 
        # But just in case:
        rows = _greedy_construct(n_rows, k)
        
    return rows

# EVOLVE-BLOCK-END
