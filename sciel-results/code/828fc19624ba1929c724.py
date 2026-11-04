"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within each row are distinct across all rows.
    Minimizes the largest entry (scope).
    
    Strategy: 
    1. Generate an initial valid configuration using a deterministic greedy algorithm.
    2. Improve it using Local Search (Hill Climbing) to minimize the maximum entry.
       - We repeatedly try to decrease random elements in the rows.
       - We bias the selection towards elements that are currently the global maximum.
       - If a decrease maintains validity (no duplicate differences), we accept it.
    """
    import time
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer

    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    lb = total_diffs

    def greedy_construct():
        """Generate a valid set using greedy approach."""
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, used

    # Step 1: Get initial valid solution
    rows, used_diffs = greedy_construct()
    
    # Step 2: Local Search to minimize scope
    current_scope = max(max(row) for row in rows)
    row_maxes = [max(row) for row in rows]
    
    # Cache the list of indices that hold the current global max
    # We will update this occasionally
    
    while time.time() - start_time < time_limit:
        # Determine current global max and indices
        # Doing this every iteration is O(n_rows * k), which is fast for small k.
        # But we can be smarter.
        current_max_val = max(row_maxes)
        
        # Find all (r, idx) where rows[r][idx] == current_max_val
        max_indices = []
        for r in range(n_rows):
            for idx in range(k + 1):
                if rows[r][idx] == current_max_val:
                    max_indices.append((r, idx))
        
        # Bias: 80% chance to pick a max index, 20% random
        if max_indices and random.random() < 0.8:
            r, idx = random.choice(max_indices)
        else:
            r = random.randint(0, n_rows - 1)
            idx = random.randint(1, k) # idx 0 is always 0, cannot change
        
        old_val = rows[r][idx]
        min_val = rows[r][idx - 1] + 1
        
        if old_val <= min_val:
            continue
            
        # Try to decrease old_val
        # Try a few random candidates in [min_val, old_val - 1]
        # If the range is small, try all. If large, try a few.
        range_size = old_val - min_val
        if range_size <= 5:
            candidates = list(range(min_val, old_val))
            random.shuffle(candidates)
        else:
            # Try 5 random candidates
            candidates = [random.randint(min_val, old_val - 1) for _ in range(5)]
            
        for new_val in candidates:
            if new_val <= rows[r][idx - 1]:
                continue
            if idx < k and new_val >= rows[r][idx + 1]:
                continue
                
            # Check validity
            removed = set()
            added = set()
            
            for j in range(k + 1):
                if j == idx:
                    continue
                if j < idx:
                    removed.add(old_val - rows[r][j])
                    added.add(new_val - rows[r][j])
                else: # j > idx
                    removed.add(rows[r][j] - old_val)
                    added.add(rows[r][j] - new_val)
            
            # Check if added conflicts with (used_diffs - removed)
            conflict = False
            for d in added:
                if d in used_diffs and d not in removed:
                    conflict = True
                    break
            
            if not conflict:
                # Accept move
                used_diffs -= removed
                used_diffs |= added
                rows[r][idx] = new_val
                
                # Update row max
                if old_val == row_maxes[r]:
                    # Max might have decreased
                    row_maxes[r] = max(rows[r])
                # If old_val was not max, max is unchanged (since new_val < old_val)
                
                break # Move accepted, try next iteration of main loop

    return rows

# EVOLVE-BLOCK-END
