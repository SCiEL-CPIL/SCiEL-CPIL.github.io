"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows of k+1 increasing integers
    starting at 0, such that all positive differences within rows are distinct across
    all rows. Minimize the largest entry.
    
    Uses a systematic greedy construction with look-ahead and iterative improvement.
    """
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = (total_diffs + n_rows - 1) // n_rows
    
    start_time = time.time()
    time_limit = 115
    
    def is_valid_solution(rows):
        """Check if solution is valid and return max value."""
        all_diffs = set()
        max_val = 0
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return None
            for i in range(len(row)):
                if i > 0 and row[i] <= row[i-1]:
                    return None
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return None
                    all_diffs.add(d)
                max_val = max(max_val, row[i])
        return max_val
    
    def greedy_construct(n_rows, k, max_val_limit, seed=0):
        """Greedy construction with look-ahead."""
        rng = random.Random(seed)
        used_diffs = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            for col in range(1, k + 1):
                min_val = row[-1] + 1
                found = False
                
                # Try values from min_val to max_val_limit
                # With look-ahead: prefer values that leave more options for future
                candidates = list(range(min_val, max_val_limit + 1))
                
                # Heuristic: sort by number of conflicts with future needs
                # For now, try in order but with some randomization
                if seed % 3 == 0:
                    rng.shuffle(candidates)
                elif seed % 3 == 1:
                    # Try smaller values first
                    pass
                else:
                    # Try values spread out
                    step = max(1, len(candidates) // 10)
                    candidates = [candidates[i] for i in range(0, len(candidates), step)] + \
                                 [candidates[i] for i in range(len(candidates) % step, len(candidates), step)]
                
                for x in candidates:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        # Forward check: can we complete this row?
                        # Simple check: ensure enough room
                        remaining_cols = k - col
                        if max_val_limit - x >= remaining_cols:
                            row.append(x)
                            used_diffs |= new_diffs
                            found = True
                            break
                
                if not found:
                    return None
            
            rows.append(row)
        
        return rows
    
    def local_improve(rows, used_diffs, max_iterations=10000):
        """Try to improve solution by reducing max value."""
        current_max = max(max(row) for row in rows)
        rng = random.Random(42)
        
        for _ in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random row and position to try to reduce
            row_idx = rng.randrange(n_rows)
            col_idx = rng.randrange(1, k + 1)  # Don't touch position 0
            
            old_val = rows[row_idx][col_idx]
            
            # Try to reduce this value
            for new_val in range(rows[row_idx][col_idx-1] + 1, old_val):
                # Check if reducing this value causes conflicts
                new_diffs = set()
                conflicts = False
                
                # Remove old differences involving this position
                old_diffs = set()
                for i in range(k + 1):
                    if i != col_idx:
                        d = abs(rows[row_idx][col_idx] - rows[row_idx][i])
                        old_diffs.add(d)
                
                # Check new value
                for i in range(k + 1):
                    if i != col_idx:
                        d = abs(new_val - rows[row_idx][i])
                        if d in used_diffs and d not in old_diffs:
                            conflicts = True
                            break
                        new_diffs.add(d)
                
                if not conflicts:
                    # Update
                    used_diffs -= old_diffs
                    used_diffs |= new_diffs
                    rows[row_idx][col_idx] = new_val
                    break
        
        return rows, current_max
    
    # Try multiple strategies
    best_solution = None
    best_max = float('inf')
    
    # Strategy 1: Systematic greedy with increasing max_val
    for offset in range(0, 30):
        max_val_limit = lower_bound + offset
        for seed in range(5):
            if time.time() - start_time > time_limit:
                break
            solution = greedy_construct(n_rows, k, max_val_limit, seed)
            if solution is not None:
                max_val = is_valid_solution(solution)
                if max_val is not None and max_val < best_max:
                    best_max = max_val
                    best_solution = [list(r) for r in solution]
                    if best_max <= lower_bound:
                        return best_solution
    
    # Strategy 2: Start with a feasible solution and improve
    if best_solution is None:
        # Use a larger bound to get a feasible solution
        for max_val_limit in range(lower_bound + 30, lower_bound + 80):
            if time.time() - start_time > time_limit:
                break
            solution = greedy_construct(n_rows, k, max_val_limit, seed=0)
            if solution is not None:
                max_val = is_valid_solution(solution)
                if max_val is not None:
                    best_max = max_val
                    best_solution = [list(r) for r in solution]
                    break
    
    if best_solution is not None:
        return best_solution
    
    # Final fallback: pure greedy
    used, rows = set(), []
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

# EVOLVE-BLOCK-END
