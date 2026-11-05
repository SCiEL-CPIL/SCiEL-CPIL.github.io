import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on max entry
    lower_bound = total_diffs
    
    # Helper to get max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)

    # Initial greedy baseline (unbounded)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    best_rows = [row[:] for row in rows]
    best_max = get_max_entry(best_rows)
    
    # Randomized Greedy Constructor with Bound M
    def randomized_greedy_with_bound(M):
        """
        Attempts to construct n_rows valid rows with all entries <= M.
        Uses randomization to escape local minima.
        Returns the rows if successful, None otherwise.
        """
        used = set()
        rows = []
        
        for r in range(n_rows):
            row = [0]
            x = 0
            # We will pick the next element.
            # To add randomness, we can consider a window of candidates [x+1, x+window_size]
            # and pick a valid one randomly. If none, expand window.
            # Or simply pick the smallest valid, but with some probability pick a larger valid one?
            # Smallest valid is best for minimizing max.
            # So we stick to smallest valid, but maybe randomize the order of rows?
            # Rows are independent given 'used'.
            # Let's try a different randomization: 
            # Instead of always picking the smallest x, pick a random x from [1, M] that is valid?
            # No, that might lead to large x quickly.
            
            # Let's stick to "smallest valid" but if we get stuck, backtrack?
            # Backtracking is expensive.
            
            # Alternative: Randomized Greedy from the end?
            # No.
            
            # Let's use a simple randomization: 
            # When choosing the next element, consider all valid x in [x+1, M].
            # If there are many, pick one randomly? 
            # This might increase the max entry.
            
            # Better: Pick the smallest valid x. This is deterministic.
            # To make it randomized, we can shuffle the rows? No, rows are symmetric.
            # We can start the search for x from a random offset? No.
            
            # Let's try this: 
            # Generate a random permutation of the "slots" in the row? No.
            
            # Let's just use the deterministic greedy but call it multiple times? 
            # It's the same result every time.
            
            # We need a better randomization.
            # Idea: When picking the next element, look at the next W candidates.
            # If any are valid, pick one randomly from the valid ones.
            # W starts small (e.g., 1) and increases if we get stuck?
            
            W = 5
            while len(row) < k + 1:
                x += 1
                if x > M:
                    return None
                
                # Check candidates x, x+1, ..., x+W-1
                candidates = []
                for candidate in range(x, min(x + W, M + 1)):
                    new_diffs = {candidate - a for a in row}
                    if not new_diffs & used:
                        candidates.append(candidate)
                
                if not candidates:
                    # No valid candidate in window. 
                    # Expand window or jump?
                    # If we jump, we might miss a smaller one.
                    # Let's just continue to x+1.
                    continue
                
                # Pick a random candidate
                # Bias towards smaller ones? 
                # Pick the smallest 50% of the time?
                if random.random() < 0.5:
                    chosen = candidates[0]
                else:
                    chosen = random.choice(candidates)
                
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
                x = chosen # Next x must be > chosen
                
        return rows

    # Search for minimal M
    current_M = lower_bound
    
    while current_M < best_max and time.time() < deadline:
        # Time budget for this M
        start_attempt_time = time.time()
        found = False
        
        while time.time() - start_attempt_time < 1.0: # 1 second per M
            rows = randomized_greedy_with_bound(current_M)
            if rows:
                found = True
                best_rows = rows
                best_max = current_M
                break
        
        if found:
            # Since we are iterating up from LB, this is the smallest M we found.
            # We can stop.
            break
        else:
            current_M += 1
            
    return best_rows
