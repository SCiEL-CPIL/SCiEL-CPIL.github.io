import time
import random

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Initial Greedy Construction
    # This provides a valid baseline.
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
    
    best_scope = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    
    # Helper to check if a move is valid and update state
    # Returns True if valid, False otherwise.
    # Modifies 'used' and 'rows' in place if valid.
    def try_move(r_idx, m_idx, new_val):
        nonlocal used
        row = rows[r_idx]
        old_val = row[m_idx]
        if new_val <= row[m_idx-1] or (m_idx < k and new_val >= row[m_idx+1]):
            return False
        
        # Calculate diffs to remove
        to_remove = set()
        for i in range(m_idx):
            to_remove.add(old_val - row[i])
        for i in range(m_idx + 1, k + 1):
            to_remove.add(row[i] - old_val)
        
        # Calculate diffs to add
        to_add = set()
        for i in range(m_idx):
            to_add.add(new_val - row[i])
        for i in range(m_idx + 1, k + 1):
            to_add.add(row[i] - new_val)
        
        # Check for collisions
        # 1. Internal collisions within the new row involving the moved element.
        # The number of pairs involving index m_idx is k.
        # If len(to_add) < k, there is a collision (e.g., new_val - row[i] == row[j] - new_val).
        if len(to_add) != k:
            return False

        # 2. Collisions with other existing differences.
        # The set of currently used diffs is 'used'.
        # We are removing 'to_remove'. The remaining used diffs are 'used' - 'to_remove'.
        # We need 'to_add' to be disjoint from 'used' - 'to_remove'.
        
        remaining_used = used - to_remove
        if to_add & remaining_used:
            return False
            
        # If valid, apply
        used -= to_remove
        used |= to_add
        row[m_idx] = new_val
        return True

    def revert_move(r_idx, m_idx, new_val, old_val):
        nonlocal used
        row = rows[r_idx]
        # This is the inverse of try_move
        to_remove = set()
        for i in range(m_idx):
            to_remove.add(new_val - row[i])
        for i in range(m_idx + 1, k + 1):
            to_remove.add(row[i] - new_val)
            
        to_add = set()
        for i in range(m_idx):
            to_add.add(old_val - row[i])
        for i in range(m_idx + 1, k + 1):
            to_add.add(row[i] - old_val)
            
        used -= to_remove
        used |= to_add
        row[m_idx] = old_val

    # 2. Local Search / Reduction
    # We try to reduce the maximum scope.
    # Strategy:
    # 1. Find the row with the current max scope.
    # 2. Try to reduce its last element.
    # 3. If stuck, perform random perturbations to escape local minima.
    
    # We keep a global 'used' set consistent with 'rows'.
    # Note: 'used' from greedy init is already correct.
    
    current_scope = best_scope
    
    while time.time() < deadline:
        # Find the row with the maximum last element
        max_idx = -1
        max_val = -1
        for i in range(n_rows):
            if rows[i][-1] > max_val:
                max_val = rows[i][-1]
                max_idx = i
        
        if max_val < best_scope:
            best_scope = max_val
            best_rows = [r[:] for r in rows]
            
        # Try to reduce the last mark of the row with max scope
        reduced = False
        r = max_idx
        last_mark_idx = k
        current_last = rows[r][-1]
        
        # Try decreasing the last mark
        for new_val in range(current_last - 1, rows[r][-2], -1):
            if try_move(r, last_mark_idx, new_val):
                reduced = True
                break
        
        if reduced:
            # If we reduced the max, the max might still be in this row or another.
            # We just loop back to find the new max.
            continue
        
        # If we couldn't reduce the max directly, try to perturb other marks
        # to create space or resolve conflicts.
        # Pick a random row and a random non-last mark (or even last mark of a smaller row)
        # and try to move it to a random smaller position.
        
        # To be more effective, try to move marks that are "blocking" the reduction.
        # But for simplicity, random perturbation often works.
        
        # Let's try to move a mark in the max row (other than the last one) to a smaller value.
        # This might free up differences that were preventing the last mark from moving down.
        
        perturbed = False
        for _ in range(100): # Try a few perturbations per stuck step
            if time.time() > deadline:
                break
            r_rand = random.randint(0, n_rows - 1)
            m_rand = random.randint(1, k - 1) # Don't move the first mark (0) or the last one usually
            
            old_val = rows[r_rand][m_rand]
            # Try to move to a random smaller valid position
            min_val = rows[r_rand][m_rand-1] + 1
            max_val_move = rows[r_rand][m_rand+1] - 1 if m_rand < k else old_val - 1
            
            if max_val_move < min_val:
                continue
                
            new_val = random.randint(min_val, max_val_move)
            if new_val == old_val:
                continue
                
            if try_move(r_rand, m_rand, new_val):
                perturbed = True
                # If we perturbed, we might be able to reduce the max now.
                # But we should also consider if this perturbation increased the scope?
                # No, we only move to smaller values, so scope never increases.
                # So this is safe.
                break
        
        if not perturbed:
            # Try moving the last mark of a non-max row?
            # Or just restart?
            # Let's try moving the last mark of a random row to a smaller value.
            r_rand = random.randint(0, n_rows - 1)
            m_rand = k
            old_val = rows[r_rand][m_rand]
            min_val = rows[r_rand][m_rand-1] + 1
            max_val_move = old_val - 1
            
            if max_val_move >= min_val:
                new_val = random.randint(min_val, max_val_move)
                if try_move(r_rand, m_rand, new_val):
                    perturbed = True

        if not perturbed:
            # If we can't do anything, we might be in a dead end for this configuration.
            # We can try a more drastic reset: re-greedy one row?
            # Or just break and restart the whole search with a different random seed?
            # Given the time limit, restarting the whole greedy+search is expensive.
            # Let's try to "shake" the solution by moving a mark to a random LARGER value?
            # No, that increases scope.
            
            # If we are stuck, we can try to swap the order of rows? No, rows are independent.
            
            # Let's try to move a mark to a random value (larger or smaller) to escape.
            # But larger is bad for score.
            
            # If we are truly stuck, we can terminate early if we've been stuck for a long time.
            # For now, we just loop. The random choices will eventually change.
            pass

    return best_rows
