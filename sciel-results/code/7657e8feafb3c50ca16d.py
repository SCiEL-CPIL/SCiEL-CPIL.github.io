"""Difference triangle set construction via randomized backtracking search."""
import random
import time
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    # Total number of differences needed: n_rows * (k+1)*k/2
    total_diffs = n_rows * (k + 1) * k // 2
    
    # Lower bound for M: M >= total_diffs
    # Because all differences must be distinct positive integers <= M.
    # There are M such integers. We need total_diffs of them.
    M_lower = total_diffs
    
    start_time = time.time()
    time_limit = 110.0
    
    def solve_for_M(M, seed):
        """
        Try to find a valid configuration with max entry <= M.
        Returns the list of rows if found, None otherwise.
        """
        rng = random.Random(seed)
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, pos, current_row):
            # Check time limit
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == 0:
                current_row = [0]
                return backtrack(row_idx, 1, current_row)
            
            if pos == k + 1:
                # Row is complete
                rows.append(list(current_row))
                return backtrack(row_idx + 1, 0, [])
            
            # We need to place current_row[pos]
            min_val = current_row[pos - 1] + 1
            max_val = M
            
            # Heuristic: Try values in random order to explore different branches
            candidates = list(range(min_val, max_val + 1))
            rng.shuffle(candidates)
            
            for x in candidates:
                # Calculate new differences with previous elements in current_row
                new_diffs = set()
                valid = True
                for i in range(pos):
                    d = x - current_row[i]
                    if d in new_diffs:
                        valid = False
                        break
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Place x
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, pos + 1, current_row):
                    return True
                
                # Unplace x
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        # Start backtracking
        if backtrack(0, 0, []):
            return rows
        return None

    # Linear search for M starting from lower bound
    current_M = M_lower
    
    # We will try a few random seeds for each M to increase probability of finding a solution
    seed_counter = 0
    
    while time.time() - start_time < time_limit:
        found = None
        # Try up to 5 seeds for this M
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            seed_counter += 1
            found = solve_for_M(current_M, seed_counter)
            if found:
                break
        
        if found:
            return found
        
        current_M += 1
        
        # If M gets very large, we might still not find a solution quickly.
        # The loop will continue until time runs out.
        
    # Fallback: Greedy construction
    # This ensures we return a valid result even if the exact search fails to find one in time.
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and len(new) == len(row):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    return greedy_construct()
