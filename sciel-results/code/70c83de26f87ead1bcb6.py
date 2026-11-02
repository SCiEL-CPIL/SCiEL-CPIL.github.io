import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_rows = None
    best_max_val = float('inf')
    
    def greedy_init(randomize=False, window_size=5):
        """
        Generates a valid set of rows using a greedy approach.
        If randomize is True, it picks randomly from the first 'window_size' valid candidates.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # We need k more numbers
            while len(row) < k + 1:
                # Start from the last number + 1
                start_x = row[-1] + 1
                candidates = []
                x = start_x
                # Look for valid candidates
                # We don't want to search forever, but greedy usually finds small ones quickly
                # However, if randomizing, we need a list of valid ones
                while len(candidates) < window_size:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        if not randomize:
                            break # If not randomizing, take the first one
                    x += 1
                    # Safety: don't search too far if something is wrong, but greedy should work
                    if x > 1000: break
                
                if not candidates:
                    # This shouldn't happen if the space is large enough, 
                    # but if it does, we have a problem. 
                    # For this task, solutions exist with M~112, so 1000 is plenty.
                    return None
                
                if randomize:
                    next_val = random.choice(candidates)
                else:
                    next_val = candidates[0]
                
                row.append(next_val)
                # Update used diffs
                used.update({next_val - a for a in row[:-1]})
            
            rows.append(row)
        return rows

    def get_max_val(rows):
        return max(max(row) for row in rows)

    def is_valid_move(rows, used, r, j, new_val):
        """
        Checks if changing rows[r][j] to new_val is valid.
        Returns True if valid.
        """
        # Check increasing order
        if j > 0 and new_val <= rows[r][j-1]:
            return False
        if j < len(rows[r]) - 1 and new_val >= rows[r][j+1]:
            return False
        
        old_val = rows[r][j]
        if new_val == old_val:
            return True # No change, technically valid but useless
        
        # Calculate old diffs to remove
        old_diffs = set()
        for i in range(len(rows[r])):
            if i == j: continue
            if i < j:
                old_diffs.add(old_val - rows[r][i])
            else:
                old_diffs.add(rows[r][i] - old_val)
        
        # Calculate new diffs to add
        new_diffs = set()
        for i in range(len(rows[r])):
            if i == j: continue
            if i < j:
                new_diffs.add(new_val - rows[r][i])
            else:
                new_diffs.add(rows[r][i] - new_val)
        
        # Check conflicts
        # The new diffs must not be in 'used' after removing 'old_diffs'
        # i.e., new_diffs & (used - old_diffs) must be empty
        
        # Optimization: 
        # If d in new_diffs:
        #   if d in old_diffs: continue (it's being freed)
        #   if d in used: return False
        
        for d in new_diffs:
            if d in old_diffs:
                continue
            if d in used:
                return False
        return True

    def apply_move(rows, used, r, j, new_val):
        """
        Applies the move and updates used.
        """
        old_val = rows[r][j]
        if new_val == old_val:
            return
        
        # Remove old diffs
        for i in range(len(rows[r])):
            if i == j: continue
            if i < j:
                used.discard(old_val - rows[r][i])
            else:
                used.discard(rows[r][i] - old_val)
        
        # Add new diffs
        for i in range(len(rows[r])):
            if i == j: continue
            if i < j:
                used.add(new_val - rows[r][i])
            else:
                used.add(rows[r][i] - new_val)
        
        rows[r][j] = new_val

    def local_search(rows, used):
        """
        Hill climbing to reduce the maximum value.
        """
        improved = True
        while improved:
            improved = False
            # Find the global max
            max_val = 0
            max_r = -1
            max_j = -1
            for r in range(len(rows)):
                for j in range(len(rows[r])):
                    if rows[r][j] > max_val:
                        max_val = rows[r][j]
                        max_r = r
                        max_j = j
            
            # Try to reduce the global max
            if max_r != -1:
                r = max_r
                j = max_j
                current_val = rows[r][j]
                # Try values smaller than current_val
                # Range: (rows[r][j-1], current_val)
                lower_bound = rows[r][j-1] + 1 if j > 0 else 1 # If j==0, it's 0, but we don't move 0? 
                # Actually, row[0] is always 0. We should not move index 0.
                if j == 0:
                    # Cannot move the first element (must be 0)
                    pass
                else:
                    for new_val in range(current_val - 1, lower_bound - 1, -1):
                        if is_valid_move(rows, used, r, j, new_val):
                            apply_move(rows, used, r, j, new_val)
                            improved = True
                            break # Restart search with new max
            
            # If we couldn't reduce the global max, try to reduce the next largest?
            # Or just stop? 
            # Stopping at local minimum for max reduction is a good heuristic.
            # To escape, we might need random kicks, but let's try simple hill climb first.
            
            # If no improvement was made in the global max, we are stuck.
            # We can try to reduce other large elements to "free up" space?
            # This is complex. Let's just return if no global max reduction.
            
        return rows, used

    # Main loop: Try multiple random initializations
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Increase randomization window slightly over time? Or keep it fixed.
        # Small window (2-5) keeps it close to greedy but diverse.
        rows = greedy_init(randomize=True, window_size=3)
        if rows is None:
            continue
            
        # Reconstruct used set for local search
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        
        # Local Search
        rows, used = local_search(rows, used)
        
        current_max = get_max_val(rows)
        if current_max < best_max_val:
            best_max_val = current_max
            best_rows = [row[:] for row in rows] # Deep copy
            
        # If we reached the theoretical lower bound (counting bound is 105 for 7,6?), 
        # we can stop early if we hit it? 
        # The counting bound for n=7, k=6 (7 rows of 7 elements? No, "7 rows of 6 increasing integers starting at 0" -> k=6 means 7 elements? 
        # Task says: "Choose 7 rows of 6 increasing integers starting at 0" -> This implies k=5? 
        # "list of k + 1 integers". 
        # If "6 increasing integers", then k+1=6 => k=5.
        # If "7 rows of 6...", usually means 6 numbers.
        # Let's check the seed: `while len(row) < k + 1`.
        # If k=5, row has 6 elements.
        # If the task says "6 increasing integers", k is likely 5.
        # The prompt says: "Choose 7 rows of 6 increasing integers...". 
        # So k=5.
        # The evaluator passes n_rows and k.
        # We don't know k here, it's an argument.
        
        # If best_max_val is very low, we might stop.
        if best_max_val <= 105: # Just a heuristic lower bound guess
             break

    if best_rows is None:
        # Fallback to pure greedy
        best_rows = greedy_init(randomize=False)
        
    return best_rows
# EVOLVE-BLOCK-END
