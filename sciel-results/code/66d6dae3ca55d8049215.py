import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct a difference triangle set for n_rows rows of k+1 integers.
    Returns a list of rows.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate initial valid solution using greedy
    initial_rows = _greedy_construct(n_rows, k)
    best_score = _get_scope(initial_rows)
    best_rows = [row[:] for row in initial_rows]
    
    # Precompute differences for all rows
    current_rows = [row[:] for row in initial_rows]
    row_diffs = [_get_row_diffs(r) for r in current_rows]
    current_score = _get_scope(current_rows)
    
    # 2. Iterated Local Search with Row Reconstruction
    # Strategy: 
    # - Pick a random row.
    # - Try to reconstruct it with a smaller max value using randomized greedy, 
    #   avoiding conflicts with other rows.
    # - If successful and improves global scope, accept.
    # - If stuck, perform a random perturbation (swap or randomize a row) and continue.
    
    # We will use a simple simulated annealing-like acceptance for row swaps,
    # but focus heavily on the "Reconstruction" move which is structurally different
    # from simple coordinate descent.
    
    T = 10.0
    cooling_rate = 0.995
    T_min = 0.1
    no_improve_count = 0
    max_no_improve = 50
    
    # Helper to check if a new row is valid against all other rows
    def is_valid_row(new_row, other_diffs):
        new_diffs = _get_row_diffs(new_row)
        for od in other_diffs:
            if new_diffs & od:
                return False
        return True
    
    # Helper to reconstruct a row given a set of forbidden differences and a target max
    def reconstruct_row(forbidden_diffs, target_max, k, rng):
        """
        Attempts to build a row of length k+1 starting at 0, strictly increasing,
        with max <= target_max, such that its differences do not intersect forbidden_diffs.
        Uses randomized backtracking/greedy.
        """
        # We want to find a set {0, a1, ..., ak} with ak <= target_max
        # such that all differences are unique and not in forbidden_diffs.
        
        # Greedy with randomization:
        # Try to pick a_i from range [a_{i-1}+1, target_max]
        # such that {a_i - a_j for j < i} is disjoint from forbidden_diffs and previously used diffs in this row.
        
        row = [0]
        used_in_row = set()
        
        for i in range(1, k + 1):
            candidates = []
            # Search range
            # Lower bound: row[-1] + 1
            # Upper bound: target_max
            lo = row[-1] + 1
            hi = target_max
            
            # To speed up, we can sample candidates or scan
            # For small k and target_max, scanning is fine.
            # But if target_max is large, it might be slow.
            # Let's scan, but limit the number of attempts if we want speed.
            
            # Actually, just scan from lo to hi.
            # If we find a valid one, we can either take it immediately (greedy) 
            # or collect a few and pick randomly (randomized greedy).
            # Randomized greedy helps escape local minima in the row structure.
            
            found = []
            # To avoid O(N^2) in worst case if many candidates, we can just take the first few valid ones
            # or pick one at random from a subset.
            # Let's just scan and break early if we find enough candidates to choose from.
            
            for x in range(lo, hi + 1):
                new_diffs = {x - a for a in row}
                # Check if new_diffs are valid
                # 1. Distinct within new_diffs? (Implied if x - a_i != x - a_j for i!=j, which is true since a_i != a_j)
                # 2. Disjoint from used_in_row?
                # 3. Disjoint from forbidden_diffs?
                
                if new_diffs & used_in_row:
                    continue
                if new_diffs & forbidden_diffs:
                    continue
                
                # Valid candidate
                found.append(x)
                if len(found) >= 10:  # Cap candidates for speed and diversity
                    break
            
            if not found:
                return None
            
            # Pick randomly from found
            x = rng.choice(found)
            row.append(x)
            new_diffs = {x - a for a in row[:-1]}
            used_in_row |= new_diffs
            
            # If we've reached the last element, we must ensure it is exactly <= target_max (which it is)
            # But wait, the last element MUST be <= target_max.
            # If i == k, we are done.
            
        return row

    rng = random.Random()
    
    while time.time() < deadline:
        # Pick a random row to modify
        idx = rng.randint(0, n_rows - 1)
        
        # Calculate forbidden differences from all OTHER rows
        # Optimization: We can maintain a global set of diffs, but removing one row's diffs is O(N).
        # Since n_rows is small (7), just build the set each time or maintain a list of sets.
        forbidden = set()
        for j in range(n_rows):
            if j != idx:
                forbidden |= row_diffs[j]
        
        current_row = current_rows[idx]
        current_max = current_row[-1]
        
        # Target: Try to find a row with max < current_max
        # Or maybe max <= current_max - 1
        # Let's try a range of target maxes, starting from current_max - 1 downwards?
        # Or just try current_max - 1 first. If it fails, try a random lower bound?
        # Actually, if we can find ANY valid row with max < current_max, the global scope might decrease.
        
        # Let's try to reconstruct with target_max = current_max - 1
        # If that fails, try a random target_max in [current_max - 10, current_max - 1]
        # If that fails, try a random target_max in [0, current_max - 1]
        
        success = False
        new_row = None
        
        # Attempt 1: Target current_max - 1
        if current_max > 0:
            new_row = reconstruct_row(forbidden, current_max - 1, k, rng)
            if new_row is not None:
                success = True
        
        # Attempt 2: Random lower target
        if not success and current_max > 5:
            target = rng.randint(max(1, current_max - 20), current_max - 1)
            new_row = reconstruct_row(forbidden, target, k, rng)
            if new_row is not None:
                success = True
        
        # Attempt 3: Any valid row (might have same or higher max, but different structure)
        # This is useful for perturbation.
        if not success:
            # Try to find a valid row with max <= current_max + 10 (slight expansion)
            # This allows "moves" that don't immediately improve scope but might enable future improvements.
            target = current_max + 10
            new_row = reconstruct_row(forbidden, target, k, rng)
            if new_row is not None:
                # This is a "neutral" or "worse" move structurally, but valid.
                # We might accept it with low probability to explore.
                pass
            else:
                # If we can't even find a valid row, we are stuck.
                # Try a random perturbation: pick a random valid row from a pool?
                # Or just break and restart?
                # For now, if we can't find a valid row, we might need to change a DIFFERENT row first.
                # So just continue the loop.
                continue

        if new_row is None:
            continue
            
        # Evaluate the new configuration
        new_row_max = new_row[-1]
        
        # Calculate new global scope
        # The global scope is max of all row maxes.
        # We are replacing row idx.
        # Other maxes are fixed.
        other_maxes = [current_rows[j][-1] for j in range(n_rows) if j != idx]
        other_max = max(other_maxes) if other_maxes else 0
        
        new_global_scope = max(new_row_max, other_max)
        
        delta = new_global_scope - current_score
        
        # Acceptance
        accept = False
        if delta < 0:
            accept = True
        else:
            # Accept worse moves with probability exp(-delta/T)
            if T > T_min:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            # Update state
            current_rows[idx] = new_row
            row_diffs[idx] = _get_row_diffs(new_row)
            current_score = new_global_scope
            
            if current_score < best_score:
                best_score = current_score
                best_rows = [row[:] for row in current_rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Cooling
            T = max(T_min, T * cooling_rate)
            
            # If we improved, maybe cool faster?
            if delta < 0:
                T = max(T_min, T * 0.95)
                
        else:
            no_improve_count += 1
            T = max(T_min, T * cooling_rate)
            
        # If stuck for too long, restart T or do a stronger perturbation
        if no_improve_count > max_no_improve:
            T = 10.0
            no_improve_count = 0
            # Optional: Randomly shuffle rows or pick a random row to "destroy" and rebuild?
            # For now, just reset temperature.
            
    return best_rows

def _greedy_construct(n_rows, k):
    """
    Greedy construction of rows.
    """
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
                used |= new_diffs
        rows.append(row)
    return rows

def _get_scope(rows):
    """
    Returns the maximum entry in any row.
    """
    scope = 0
    for row in rows:
        if row:
            scope = max(scope, row[-1])
    return scope

def _get_row_diffs(row):
    """
    Compute the set of all positive differences in a row.
    """
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

# EVOLVE-BLOCK-END
