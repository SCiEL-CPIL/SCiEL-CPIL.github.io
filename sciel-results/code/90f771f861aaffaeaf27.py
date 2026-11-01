"""
Strategy: Local Search / Hill Climbing for Scope Minimization
Instead of constructing rows from scratch (which is prone to getting stuck in local minima
due to the sequential nature of the greedy choice), we start with a valid solution and
iteratively improve it by moving individual elements to smaller values.

The "scope" is the maximum entry across all rows. We aim to minimize this.
We represent the solution as a list of rows.
A "move" is changing a specific element a[i][j] to a smaller value v.
Validity constraints:
1. Row must remain strictly increasing.
2. All pairwise differences within a row must be distinct.
3. No difference from any row can match a difference from any other row.

We use a randomized local search:
1. Generate an initial valid solution (using a simple greedy or randomized construction).
2. Repeatedly try to reduce the scope:
   a. Identify the "bottleneck" rows (those with max entry close to current scope).
   b. For elements in these rows, try to move them to smaller values.
   c. If a move reduces the max entry of the row without violating constraints, accept it.
   d. Use simulated annealing or random restarts to escape local minima where no single
      element move improves the scope, but a sequence of moves might.

Key Insight: The problem is small (7 rows, 6 elements). The constraint check is O(N^2 * K^2) 
which is very fast. We can afford many iterations.
"""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    deadline = time.time() + 145.0  # 5s safety margin
    
    # 1. Generate an initial valid solution
    # We use a simple greedy construction to get a valid start.
    # This might not be optimal, but it's valid.
    initial_rows = _greedy_construct(n_rows, k)
    best_rows = initial_rows
    best_scope = _max_entry(initial_rows)
    
    # If the problem is trivial, return immediately
    if n_rows <= 1 or k <= 1:
        return best_rows

    # 2. Local Search / Hill Climbing
    # We will run multiple restarts.
    # In each restart, we try to improve the scope.
    
    # Pre-compute all possible differences? 
    # Max difference is bounded by scope. Scope starts around 100-120.
    # We can use a set for global used differences.
    
    num_restarts = 1000 # We will break early based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # For each restart, we start from the current best or a perturbed version?
        # Starting from the best is good for exploitation.
        # To explore, we could perturb the best solution.
        # Let's try starting from the best solution and doing a random walk of moves.
        
        current_rows = [row[:] for row in best_rows]
        current_scope = best_scope
        
        # We will try to improve the scope.
        # Strategy: 
        # While time remains:
        #   Pick a random row.
        #   Pick a random index in that row (except index 0, which is fixed to 0).
        #   Try to decrease the value at that index.
        #   If the new value is valid (row increasing, diffs unique globally), accept.
        #   If the new value reduces the max entry of the row, and that max entry was the global max,
        #     we might have improved the global scope.
        
        # To be more effective, we should focus on the rows that determine the global max.
        
        # Let's implement a specific "scope reduction" heuristic.
        # 1. Find the current global max value M and which rows have it.
        # 2. For each row with max entry M, try to reduce the last element (index k) to M-1, M-2, etc.
        # 3. If that fails, try to reduce the second-to-last element, etc.
        # 4. If we successfully reduce a row's max entry below M, the global scope might drop.
        
        # However, simply reducing one element might break the "strictly increasing" or "diff uniqueness".
        # We need a robust validity check.
        
        # Let's use a "randomized hill climbing" with a focus on the bottleneck.
        
        # We'll maintain a set of used differences for fast checking.
        used_diffs = _compute_used_diffs(current_rows)
        
        # We will do a series of moves.
        # Move: change current_rows[r][j] to a new value v.
        # Constraints:
        # - v > current_rows[r][j-1] (if j>0)
        # - v < current_rows[r][j+1] (if j<k)
        # - New differences {v - current_rows[r][i] for i != j} must not conflict with 
        #   existing differences in the row (excluding those involving j) 
        #   and must not conflict with differences in other rows.
        
        # To make this efficient, we can pre-compute the "forbidden" values for each position?
        # That might be too heavy. Let's just check on the fly.
        
        # We will iterate for a fixed number of steps per restart, or until no improvement.
        # Since we want to minimize the scope, we can define a "temperature" or just accept 
        # any move that doesn't increase the scope, and prefer moves that decrease it.
        
        # Actually, a simpler and often more effective approach for this specific "minimize max" problem:
        # 1. Identify the row with the maximum entry.
        # 2. Try to "shrink" that row by moving its largest element down.
        # 3. If that fails, move the second largest down, etc.
        # 4. If the row is shrunk, re-evaluate the global max.
        
        # Let's implement a "Targeted Shrinking" strategy.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Find the row with the maximum entry
            max_val = -1
            max_row_idx = -1
            for r in range(n_rows):
                if current_rows[r][-1] > max_val:
                    max_val = current_rows[r][-1]
                    max_row_idx = r
            
            # Try to shrink the max_row_idx
            # We iterate from the last element to the first (excluding index 0).
            # For each position j, we try to decrease the value.
            
            # To add randomness and escape local minima, we can pick a random row 
            # that is close to the max, or just stick to the max row.
            # Let's stick to the max row first. If stuck, try others.
            
            # Actually, let's try all rows that have max entry == max_val.
            # And maybe some rows that are close to max_val to "clear space".
            
            candidate_rows = [r for r in range(n_rows) if current_rows[r][-1] >= max_val - 5]
            random.shuffle(candidate_rows)
            
            for r in candidate_rows:
                if time.time() > deadline - 5:
                    break
                
                # Try to shrink row r
                # We go from the end of the row to the beginning.
                # Why end to beginning? Because changing an earlier element affects more differences.
                # Changing the last element only affects differences involving the last element.
                
                for j in range(k, 0, -1):
                    old_val = current_rows[r][j]
                    # Try new values from old_val - 1 down to the lower bound
                    # Lower bound: current_rows[r][j-1] + 1
                    lower_bound = current_rows[r][j-1] + 1 if j > 0 else 0
                    
                    # We want to find the smallest valid value >= lower_bound.
                    # But we also want to be careful not to break other constraints.
                    
                    # Let's try values in a randomized order or descending order?
                    # Descending (from old_val-1 down to lower_bound) allows us to find a valid 
                    # value quickly if one exists close to the current value.
                    # But if we want to minimize the scope, we want the *smallest* valid value.
                    # However, the smallest valid value might not exist.
                    # Let's try to find ANY valid value that is < old_val.
                    # If we find one, we accept it. This is a greedy step.
                    
                    # To avoid getting stuck, we can try a few random values in the range [lower_bound, old_val-1].
                    
                    found_new_val = False
                    # Try a few random candidates
                    # And also try the immediate predecessor
                    candidates = set()
                    candidates.add(old_val - 1)
                    # Add some random values
                    for _ in range(10):
                        if old_val > lower_bound + 1:
                            candidates.add(random.randint(lower_bound, old_val - 1))
                    
                    # Sort candidates to try smaller ones first? 
                    # If we try smaller ones first, we might jump to a very small value, 
                    # which might be invalid, but if valid, it's a big improvement.
                    # Let's try them in random order to explore.
                    random.shuffle(list(candidates))
                    
                    for new_val in candidates:
                        if new_val <= lower_bound or new_val >= old_val:
                            continue
                        
                        # Check validity
                        if _is_move_valid(current_rows, r, j, new_val, used_diffs):
                            # Accept move
                            # Update used_diffs
                            # Remove old diffs involving (r, j)
                            # Add new diffs involving (r, j)
                            
                            # Old diffs:
                            for i in range(k + 1):
                                if i != j:
                                    d = abs(current_rows[r][j] - current_rows[r][i])
                                    used_diffs.discard(d)
                            
                            # Update row
                            current_rows[r][j] = new_val
                            
                            # New diffs:
                            for i in range(k + 1):
                                if i != j:
                                    d = abs(new_val - current_rows[r][i])
                                    used_diffs.add(d)
                            
                            found_new_val = True
                            improved = True
                            break
                    
                    if found_new_val:
                        # If we improved, we can break out of the j loop?
                        # Or continue to shrink other elements in this row?
                        # Let's break to re-evaluate the global state.
                        break
                
                # If we improved in this row, we can break out of the row loop?
                # Or try other rows?
                # Let's break to re-evaluate max_val.
                if improved:
                    break
        
        # After the while loop, check if we improved the best
        new_scope = _max_entry(current_rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [row[:] for row in current_rows]
            
            # If we hit the theoretical lower bound, stop
            if best_scope <= 105:
                return best_rows

    return best_rows

def _is_move_valid(rows, r, j, new_val, used_diffs):
    """
    Check if setting rows[r][j] to new_val is valid.
    rows: list of rows
    r: row index
    j: column index
    new_val: new value
    used_diffs: set of all differences currently in use (including the one being replaced? 
               No, we assume the caller has not updated it yet, or we handle it carefully.
               In the main loop, we remove old diffs before calling? 
               No, in the current implementation, I didn't remove them before calling.
               So used_diffs still contains the old diffs for (r, j).
               We need to check if new_val creates any conflicts with:
               1. Other elements in the same row (excluding j).
               2. Elements in other rows.
               
               The old diffs for (r, j) are in used_diffs. 
               If a new diff matches an old diff of (r, j), that's okay because we are replacing it.
               But if a new diff matches a diff of (r, i) where i != j, that's a conflict.
               If a new diff matches a diff of another row, that's a conflict.
               
               So, we must exclude the diffs that are being replaced from the conflict check.
               Or, simpler: temporarily remove the old diffs, check, and if valid, keep the removal.
               But we don't want to modify used_diffs if the move is invalid.
               
               Let's implement a check that accounts for the replacement.
    """
    row = rows[r]
    
    # 1. Check strictly increasing
    if j > 0 and new_val <= row[j-1]:
        return False
    if j < k and new_val >= row[j+1]:
        return False
    
    # 2. Check difference uniqueness
    # New diffs are {new_val - row[i] for i != j}
    # We need to ensure:
    # a. These new diffs are distinct from each other.
    # b. These new diffs are distinct from all diffs in other rows.
    # c. These new diffs are distinct from all diffs in the same row that do NOT involve j.
    
    # Let's compute the set of "forbidden" diffs.
    # Forbidden = (All diffs in used_diffs) MINUS (Old diffs involving j in row r)
    
    old_val = row[j]
    old_diffs_to_remove = set()
    for i in range(k + 1):
        if i != j:
            d = abs(old_val - row[i])
            old_diffs_to_remove.add(d)
    
    # Forbidden diffs for the new value:
    # Any diff in used_diffs that is NOT in old_diffs_to_remove.
    # Plus, we must ensure the new diffs themselves are unique.
    
    new_diffs = set()
    for i in range(k + 1):
        if i != j:
            d = abs(new_val - row[i])
            # Check if this diff is already in new_diffs (intra-row conflict)
            if d in new_diffs:
                return False
            new_diffs.add(d)
            
            # Check if this diff conflicts with global used_diffs (excluding old_diffs_to_remove)
            if d in used_diffs and d not in old_diffs_to_remove:
                return False
    
    return True

def _compute_used_diffs(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used.add(row[j] - row[i])
    return used

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(row) for row in rows)

# EVOLVE-BLOCK-END
