"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    deadline = time.time() + 115  # Safety margin before 150s limit
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Constraint-based backtracking with iterative deepening
    # Try to find a solution with scope <= target, increasing target if needed
    
    # Lower bound: counting argument gives ~105, but let's start higher
    # Upper bound: greedy gives us a starting point
    greedy_rows = _greedy_construct(n_rows, k)
    if greedy_rows:
        best_rows = [r[:] for r in greedy_rows]
        best_scope = max(max(r) for r in best_rows)
    
    # Try backtracking with various scope limits
    # Start from a reasonable estimate and try to tighten
    target_scope = best_scope - 1
    
    while time.time() < deadline - 5 and target_scope >= 80:
        if time.time() > deadline - 10:
            break
        
        # Try to find a valid configuration with max element <= target_scope
        result = _backtrack_search(n_rows, k, target_scope, deadline)
        if result:
            scope = max(max(r) for r in result)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in result]
            # Try to go even lower
            target_scope = scope - 1
        else:
            # No solution found at this scope, try a slightly higher scope
            # But don't go above our current best
            target_scope += 1
            if target_scope >= best_scope:
                break
    
    # Strategy 2: Simulated annealing to try to improve further
    if best_rows and time.time() < deadline - 15:
        sa_result = _simulated_annealing(best_rows, k, deadline)
        if sa_result:
            sa_scope = max(max(r) for r in sa_result)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [r[:] for r in sa_result]
    
    # Strategy 3: More aggressive local search with restarts
    if best_rows and time.time() < deadline - 5:
        for restart in range(50):
            if time.time() > deadline - 3:
                break
            # Perturb the best solution
            perturbed = _perturb(best_rows, k, strength=3)
            if perturbed:
                improved, new_rows, new_scope = _local_search_v2(perturbed, k, deadline)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in new_rows]
    
    return best_rows

def _greedy_construct(n_rows, k):
    """Simple greedy: pick smallest possible next value."""
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

def _row_diffs(row):
    """Compute all pairwise differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_row_valid(row, k):
    """Check if a row has all distinct internal differences."""
    diffs = _row_diffs(row)
    return len(diffs) == k * (k + 1) // 2

def _backtrack_search(n_rows, k, max_scope, deadline):
    """
    Backtracking search for a valid configuration with max element <= max_scope.
    Uses constraint propagation and forward checking.
    """
    # All differences that must be unique: total = n_rows * k*(k+1)//2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # We'll build rows one by one
    rows = []
    used_diffs = set()
    
    def backtrack(row_idx):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        # Try to construct row row_idx
        # Start with 0
        row = [0]
        used_in_row = {0}
        
        def build_row(pos):
            if time.time() > deadline:
                return False
            
            if pos == k + 1:
                # Check row validity
                if not _is_row_valid(row, k):
                    return False
                
                # Check no conflict with used_diffs
                row_diffs = _row_diffs(row)
                if row_diffs & used_diffs:
                    return False
                
                # Accept this row
                rows.append(row[:])
                used_diffs.update(row_diffs)
                
                # Recurse
                if backtrack(row_idx + 1):
                    return True
                
                # Backtrack
                rows.pop()
                used_diffs.difference_update(row_diffs)
                return False
            
            # Try next element
            # Minimum possible value
            min_val = row[-1] + 1
            # Maximum possible value (bounded by max_scope)
            max_val = min(max_scope, row[-1] + max_scope)
            
            # Forward checking: for each candidate, check if it creates
            # a difference that's already used or would make the row invalid
            
            # Try values in increasing order (to find compact solutions)
            for cand in range(min_val, max_val + 1):
                if time.time() > deadline:
                    return False
                
                # Check internal row validity
                new_diffs = {cand - a for a in row}
                
                # Check no duplicate within the row
                if len(new_diffs) != len(row):
                    continue
                
                # Check no conflict with already used diffs
                if new_diffs & used_diffs:
                    continue
                
                # Check no conflict with diffs already in this row
                existing_row_diffs = _row_diffs(row)
                if new_diffs & existing_row_diffs:
                    continue
                
                # Accept candidate
                row.append(cand)
                if build_row(pos + 1):
                    return True
                row.pop()
            
            return False
        
        # Start building this row
        if build_row(1):
            return True
        
        return False
    
    return backtrack(0) if rows == [] else None

def _perturb(rows, k, strength=3):
    """Create a perturbed copy of rows by randomly changing some elements."""
    if not rows:
        return None
    
    perturbed = [r[:] for r in rows]
    n_rows = len(perturbed)
    
    for _ in range(strength):
        # Pick a random row and element to perturb
        row_idx = random.randint(0, n_rows - 1)
        elem_idx = random.randint(1, k)  # Skip the 0
        
        row = perturbed[row_idx]
        old_val = row[elem_idx]
        
        # Try to change to a nearby value
        delta = random.randint(-5, 5)
        new_val = old_val + delta
        
        # Check bounds
        if elem_idx > 0 and new_val <= row[elem_idx - 1]:
            new_val = row[elem_idx - 1] + 1
        if elem_idx < k and new_val >= row[elem_idx + 1]:
            new_val = row[elem_idx + 1] - 1
        
        if new_val != old_val and (elem_idx == 0 or new_val > row[elem_idx - 1]) and \
           (elem_idx == k or new_val < row[elem_idx + 1]):
            row[elem_idx] = new_val
    
    # Verify the perturbed solution is still valid
    if not _validate(perturbed, k):
        return None
    
    return perturbed

def _validate(rows, k):
    """Check if a set of rows is valid."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        # Check strictly increasing
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        
        diffs = _row_diffs(row)
        if len(diffs) != k * (k + 1) // 2:
            return False
        if diffs & used:
            return False
        used |= diffs
    return True

def _local_search_v2(rows, k, deadline):
    """
    Improved local search: try to reduce the maximum element by
    moving individual elements to smaller valid positions.
    """
    curr_rows = [r[:] for r in rows]
    n_rows = len(curr_rows)
    
    # Compute global used diffs
    global_diffs = set()
    for row in curr_rows:
        global_diffs |= _row_diffs(row)
    
    current_scope = max(max(r) for r in curr_rows)
    improved = False
    
    # Multiple passes
    max_passes = 20
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        pass_improved = False
        
        # Create list of all movable elements (row_idx, elem_idx)
        moves = []
        for i in range(n_rows):
            for j in range(1, k + 1):
                moves.append((i, j))
        random.shuffle(moves)
        
        for i, j in moves:
            if time.time() > deadline:
                return improved, curr_rows, current_scope
            
            row = curr_rows[i]
            current_val = row[j]
            
            # Determine valid range for this element
            min_val = row[j - 1] + 1
            max_val = row[j + 1] - 1 if j + 1 < len(row) else current_val
            
            # Try to find a smaller valid value
            # Scan from min_val upward to find the smallest valid value
            best_cand = None
            for cand in range(min_val, current_val):
                # Temporarily remove this row's diffs
                old_diffs = _row_diffs(row)
                global_diffs -= old_diffs
                
                # Try candidate
                new_row = row[:]
                new_row[j] = cand
                
                # Check validity
                new_diffs = _row_diffs(new_row)
                if len(new_diffs) != k * (k + 1) // 2:
                    global_diffs |= old_diffs
                    continue
                if new_diffs & global_diffs:
                    global_diffs |= old_diffs
                    continue
                
                # Valid!
                best_cand = cand
                global_diffs |= new_diffs
                break
            else:
                # No valid smaller value, restore old diffs
                global_diffs |= old_diffs
                continue
            
            # Apply the move
            row[j] = best_cand
            curr_rows[i] = row
            
            # Update scope
            new_scope = max(max(r) for r in curr_rows)
            if new_scope < current_scope:
                current_scope = new_scope
            
            improved = True
            pass_improved = True
        
        if not pass_improved:
            break
    
    return improved, curr_rows, current_scope

def _simulated_annealing(rows, k, deadline):
    """
    Simulated annealing: treat each element as a variable, minimize max element.
    """
    curr_rows = [r[:] for r in rows]
    n_rows = len(curr_rows)
    
    def compute_cost(rows):
        """Cost = max element (lower is better)."""
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        return _validate(rows, k)
    
    if not is_valid(curr_rows):
        return None
    
    current_cost = compute_cost(curr_rows)
    best_rows = [r[:] for r in curr_rows]
    best_cost = current_cost
    
    # Annealing schedule
    T = 10.0
    cooling = 0.995
    min_T = 0.01
    
    while T > min_T and time.time() < deadline:
        # Pick a random move
        row_idx = random.randint(0, n_rows - 1)
        elem_idx = random.randint(1, k)
        
        row = curr_rows[row_idx]
        old_val = row[elem_idx]
        
        # Generate a new value
        # Bias towards smaller values
        if random.random() < 0.7:
            # Move down
            delta = random.randint(1, 5)
            new_val = old_val - delta
        else:
            # Move up a bit (to escape local minima)
            delta = random.randint(1, 3)
            new_val = old_val + delta
        
        # Check bounds
        if elem_idx > 0 and new_val <= row[elem_idx - 1]:
            new_val = row[elem_idx - 1] + 1
        if elem_idx < k and new_val >= row[elem_idx + 1]:
            new_val = row[elem_idx + 1] - 1
        
        if new_val == old_val:
            continue
        
        # Apply move
        row[elem_idx] = new_val
        
        # Check validity
        if not is_valid(curr_rows):
            # Revert
            row[elem_idx] = old_val
        else:
            new_cost = compute_cost(curr_rows)
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0:
                # Accept (improvement)
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [r[:] for r in curr_rows]
            else:
                # Accept with probability exp(-delta_cost / T)
                if random.random() < pow(2.718281828, -delta_cost / T):
                    current_cost = new_cost
                else:
                    # Revert
                    row[elem_idx] = old_val
        
        T *= cooling
    
    return best_rows

# EVOLVE-BLOCK-END
