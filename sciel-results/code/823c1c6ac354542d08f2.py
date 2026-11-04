import time
import random

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True

def _greedy_random(n_rows, k, max_scope=120):
    """
    Greedy construction with randomization to find a valid set within max_scope.
    Returns None if it fails to find one within reasonable tries.
    """
    for _ in range(100):
        used = set()
        rows = []
        success = True
        for r_idx in range(n_rows):
            row = [0]
            # Try to build the rest of the row
            for pos in range(1, k + 1):
                found = False
                # Start from last element + 1, try increasing values
                # Randomize start offset to explore different structures
                start_val = row[-1] + 1
                # Add some randomness to the step size
                # We want to keep values low, so we try small steps first but with randomization
                # to avoid getting stuck in a deterministic bad path.
                
                # Heuristic: try values in a random order within a window, 
                # but bias towards smaller values to keep scope low.
                # Actually, for greedy, just picking the smallest valid is often good for scope.
                # But we want diversity. Let's try a mix.
                
                # Strategy: Try smallest valid, but if we fail to complete the row later, 
                # this specific greedy might be too tight.
                # Let's use a randomized greedy: pick a random valid candidate from a range.
                
                # To ensure we can finish, we should leave room.
                # Max remaining elements: k - pos
                # Min step 1. So we need at least k-pos more.
                # Let's just try the smallest valid value. It's the most "compact".
                # If this fails to produce a valid set of n_rows, we restart.
                
                x = start_val
                while x <= max_scope:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                    x += 1
                if not found:
                    success = False
                    break
            if not success:
                break
            rows.append(row)
        
        if success:
            # Verify
            if _valid(rows):
                return rows
    return None

def _local_search(rows, k, deadline):
    """
    Local search to minimize the scope.
    Strategy: 
    1. Maintain a global set of used differences.
    2. Repeatedly pick a random row and a random element.
    3. Try to reduce the element to the smallest possible valid value.
    4. If stuck (no element can be reduced), try a perturbation.
    """
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Build initial used set
    used = set()
    for r in best_rows:
        used |= _row_diffs(r)
        
    n_rows = len(best_rows)
    
    # Helper to check if a specific move is valid
    def try_reduce(row_idx, pos):
        """
        Tries to reduce best_rows[row_idx][pos] to the smallest possible value > prev.
        Returns True if a reduction was made.
        """
        row = best_rows[row_idx]
        if pos <= 0:
            return False
        if pos >= len(row):
            return False
            
        prev_val = row[pos - 1]
        curr_val = row[pos]
        
        # If we can't reduce, return False
        if curr_val == prev_val + 1:
            return False
            
        # Identify diffs to remove
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(curr_val - row[i])
        for i in range(pos + 1, len(row)):
            old_diffs.add(row[i] - curr_val)
        
        # Temporary removal
        used_copy = set(used)
        used_copy -= old_diffs
        
        # Try new values from prev_val + 1 up to curr_val - 1
        new_val = prev_val + 1
        while new_val < curr_val:
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - new_val)
                
            # Check conflict with remaining used
            if new_diffs & used_copy:
                new_val += 1
                continue
            
            # Valid move!
            # Apply it
            row[pos] = new_val
            used -= old_diffs
            used |= new_diffs
            return True
        
        return False

    # Helper to try a random move (perturbation)
    def try_perturb():
        """
        Picks a random element and tries to change it to a random valid value 
        (not necessarily smaller) to escape local minima.
        """
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, len(best_rows[row_idx]) - 1)
        row = best_rows[row_idx]
        curr_val = row[pos]
        prev_val = row[pos - 1]
        next_val = row[pos + 1] if pos + 1 < len(row) else float('inf')
        
        # Range for new value
        low = prev_val + 1
        high = min(curr_val + 5, next_val - 1) # Small neighborhood
        
        if low > high:
            return False
            
        # Try a few random values in this range
        candidates = list(range(low, high + 1))
        random.shuffle(candidates)
        
        for new_val in candidates:
            if new_val == curr_val:
                continue
                
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(curr_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - curr_val)
                
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - new_val)
                
            # Check conflict
            used_copy = set(used)
            used_copy -= old_diffs
            if new_diffs & used_copy:
                continue
                
            # Accept
            row[pos] = new_val
            used -= old_diffs
            used |= new_diffs
            return True
        return False

    start_time = time.time()
    stuck_count = 0
    
    while time.time() < deadline:
        if time.time() > deadline - 1.0:
            break
            
        # Try to reduce a random element
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, len(best_rows[row_idx]) - 1)
        
        if try_reduce(row_idx, pos):
            stuck_count = 0
            # Check if scope improved
            current_scope = _scope(best_rows)
            if current_scope < best_scope:
                best_scope = current_scope
        else:
            stuck_count += 1
            # If stuck for a while, try a perturbation
            if stuck_count > 10:
                if try_perturb():
                    stuck_count = 0
                else:
                    # If perturbation fails, we might be in a deep local minimum.
                    # We could restart, but that's expensive. 
                    # For now, just keep trying.
                    pass

    return best_rows, best_scope

def construct(n_rows, k):
    deadline = time.time() + 115.0
    
    # 1. Generate an initial valid set
    # We want a scope that is reasonably tight. 
    # Lower bound is 105. Let's try to find one with scope <= 115.
    initial_rows = None
    # Try greedy with different random seeds implicitly by calling multiple times
    # or just one robust greedy.
    # My _greedy_random restarts 100 times.
    
    # To be safe and fast, let's try to construct with a specific target scope.
    # If we can't find one with scope <= 112, we might have to go higher, 
    # but the local search will try to bring it down.
    
    # Let's try to find a solution with scope <= 115 first.
    initial_rows = _greedy_random(n_rows, k, max_scope=115)
    
    if initial_rows is None:
        # Fallback: allow larger scope
        initial_rows = _greedy_random(n_rows, k, max_scope=150)
        
    if initial_rows is None:
        # Emergency fallback: simple greedy without scope limit
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                x += 1
            rows.append(row)
        initial_rows = rows

    # 2. Local Search
    best_rows, best_scope = _local_search(initial_rows, k, deadline)
    
    return best_rows

# EVOLVE-BLOCK-END
