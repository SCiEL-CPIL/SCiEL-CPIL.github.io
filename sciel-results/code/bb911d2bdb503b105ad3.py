"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences a_j - a_i (i < j) inside the rows must be distinct over all rows.
    Goal: Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    # --- Helper Functions ---
    
    def get_diffs(row):
        """Calculate the set of positive differences for a sorted row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_config(rows):
        """Check if the configuration is valid (all diffs distinct across rows)."""
        all_diffs = set()
        for row in rows:
            row_diffs = get_diffs(row)
            if all_diffs & row_diffs:
                return False
            all_diffs.update(row_diffs)
        return True

    def get_scope(rows):
        """Calculate the maximum entry in all rows."""
        return max(max(row) for row in rows)

    # --- Initialization: Greedy Construction ---
    
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # Start with small increments, expand as needed
            # A simple heuristic: try to place next element as small as possible
            while len(row) < k + 1:
                # Try candidates from last element + 1 upwards
                # Optimization: limit search space for initialization
                # The max possible value for a single ruler of k marks is roughly k^2/2
                # We can search up to a reasonable bound
                found = False
                # Heuristic bound for single row greedy
                limit = max(50, k * k) 
                for x in range(row[-1] + 1, limit + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                if not found:
                    # If we hit the limit, we might need to go higher, 
                    # but for initialization, let's just pick a large random valid one or expand limit
                    # For robustness, expand limit
                    for x in range(row[-1] + 1, row[-1] + 100):
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            row.append(x)
                            used_diffs.update(new_diffs)
                            found = True
                            break
                if not found:
                    # Fallback: should not happen for small k, but just in case
                    return None
            rows.append(row)
        return rows

    # --- Local Search ---
    
    def local_search(initial_rows, max_time):
        rows = [list(r) for r in initial_rows]
        if not is_valid_config(rows):
            return initial_rows
            
        current_scope = get_scope(rows)
        best_rows = [list(r) for r in rows]
        best_scope = current_scope
        
        # Pre-calculate current diffs for fast checking
        # We need to check if a new row's diffs conflict with:
        # 1. Diffs of other rows
        # 2. Diffs of the old row (excluding those removed)
        
        row_diffs = [get_diffs(r) for r in rows]
        
        # To speed up conflict checking:
        # global_diffs = union of all row_diffs
        # When changing row i:
        # new_diffs = get_diffs(new_row)
        # conflicts = (new_diffs & global_diffs) - (new_diffs & old_diffs_i)
        # Actually, simpler:
        # other_diffs = global_diffs - row_diffs[i]
        # if new_diffs & other_diffs: invalid
        
        global_diffs = set()
        for rd in row_diffs:
            global_diffs.update(rd)
            
        def update_global_diffs(old_diffs, new_diffs):
            # Remove old, add new
            # Note: if a diff is in both old and new, it stays.
            # But we are replacing a row.
            # So: global_diffs = (global_diffs - old_diffs) | new_diffs
            global_diffs.difference_update(old_diffs)
            global_diffs.update(new_diffs)

        def try_move(row_idx, new_val, old_val):
            """
            Attempt to replace old_val in rows[row_idx] with new_val.
            Returns True if valid, and updates state if so.
            """
            nonlocal global_diffs, row_diffs, rows, current_scope, best_rows, best_scope
            
            old_row = rows[row_idx]
            # Check if new_val is already in the row
            if new_val in old_row:
                return False
            if new_val == old_val:
                return False
            
            # Create new row
            new_row = [x for x in old_row if x != old_val]
            new_row.append(new_val)
            new_row.sort()
            
            # Calculate new diffs
            new_diffs = get_diffs(new_row)
            old_diffs = row_diffs[row_idx]
            
            # Check conflicts with other rows
            # Diffs that are in new_diffs and in global_diffs but NOT in old_diffs are conflicts
            # Because diffs in old_diffs are "removed" from global context for this row's sake
            # Wait, global_diffs contains old_diffs.
            # If d is in new_diffs and d is in global_diffs:
            #   If d is in old_diffs, it's fine (it was there before, still there).
            #   If d is NOT in old_diffs, it conflicts with some other row (or was in another row).
            
            # Efficient check:
            # intersection = new_diffs & global_diffs
            # bad_intersection = intersection - old_diffs
            # if bad_intersection: invalid
            
            intersection = new_diffs & global_diffs
            if intersection - old_diffs:
                return False
            
            # Move is valid. Apply it.
            # Update global_diffs
            global_diffs.difference_update(old_diffs)
            global_diffs.update(new_diffs)
            
            # Update row data
            rows[row_idx] = new_row
            row_diffs[row_idx] = new_diffs
            
            # Update scope
            new_scope = max(max(r) for r in rows)
            if new_scope < current_scope:
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
            elif new_scope > current_scope:
                # Scope increased. We might still accept it in simulated annealing,
                # but for pure hill climbing on scope, we reject.
                # However, sometimes increasing scope allows a larger decrease later?
                # Usually for this problem, we want to strictly minimize max.
                # Let's use a simple hill climbing that only accepts if scope doesn't increase.
                # Or better: accept if scope is same or better, to allow lateral moves.
                if new_scope > current_scope:
                    # Revert
                    global_diffs.difference_update(new_diffs)
                    global_diffs.update(old_diffs)
                    rows[row_idx] = old_row
                    row_diffs[row_idx] = old_diffs
                    return False
            
            return True

        # Main loop
        while time.time() < max_time:
            # Pick a random row
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]
            
            # Pick a random element to change (exclude 0, as it's fixed)
            # We can change any non-zero element.
            # Strategy: Target the largest elements first to try to reduce scope.
            # Let's pick an element from the upper half of the row's values.
            # Or just pick a random non-zero index.
            if len(row) <= 1:
                continue
                
            # To effectively reduce the max, we should try to replace the current max
            # or other large numbers with smaller ones.
            # Let's bias the selection towards larger values in the row.
            # Get indices of non-zero elements
            non_zero_indices = [i for i, val in enumerate(row) if val != 0]
            if not non_zero_indices:
                continue
                
            # Weighted random choice: larger values have higher chance of being picked
            # Simpler: just pick a random non-zero element.
            idx = random.choice(non_zero_indices)
            old_val = row[idx]
            
            # Determine range for new_val
            # We want to try smaller values to reduce scope.
            # But we also need to explore.
            # Let's try values in [1, old_val - 1] and [old_val + 1, old_val + 10]
            # If we find a valid smaller value, scope might drop.
            
            # First, try to reduce:
            if old_val > 1:
                # Try random smaller values
                # Sort candidates? No, random is fine.
                # To be efficient, try a few.
                for _ in range(5):
                    new_val = random.randint(1, old_val - 1)
                    if try_move(row_idx, new_val, old_val):
                        break
            
            # If scope didn't decrease (or we didn't find a smaller one), try lateral or slight increase?
            # For now, stick to reduction-focused moves.
            # If stuck, we can try swapping two elements?
            # Swapping is complex with the diff constraint.
            
            # Alternative move: Swap two elements in the same row?
            # No, the row is a set. Swapping values doesn't change the set.
            # So "swap" is just replacing one value with another.
            
            # If we are stuck at a local minimum, we might need a perturbation.
            # But since we only accept non-increasing scope, we might get stuck.
            # Let's allow "sideways" moves (scope stays same) to escape plateaus.
            # The try_move above allows new_scope == current_scope.
            
            # To escape deeper, we could temporarily allow an increase.
            # But let's keep it simple first.
            
            # If no move was made in the last few iterations, we might need to restart or perturb.
            # For now, the randomization in new_val selection provides diversity.

        return best_rows

    # --- Execution ---
    
    # 1. Generate initial solution
    initial = greedy_construct()
    if initial is None:
        # Fallback if greedy fails (unlikely for small k)
        # Just return a simple valid config, even if not optimal
        # Constructing a trivial one is hard if greedy fails, 
        # but greedy rarely fails for k=7.
        # Let's try a more robust greedy or random init.
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Pick random large number
                x = random.randint(row[-1] + 1, 200)
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(new_diffs)
            rows.append(row)
        initial = rows

    # 2. Run local search
    # Reserve 5s for cleanup/return
    search_deadline = start_time + 105.0 
    result = local_search(initial, search_deadline)
    
    return result

# EVOLVE-BLOCK-END
