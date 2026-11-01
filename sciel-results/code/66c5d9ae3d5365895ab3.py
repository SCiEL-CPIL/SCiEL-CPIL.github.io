import time
import random

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    deadline = time.time() + 110
    
    # Lower bound estimation
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Generate all valid Sidon rows up to a max scope
    M_max = 115
    
    rows_data = _generate_sidon_rows(M_max, k)
    
    # Sort rows by max value
    rows_data.sort(key=lambda x: x[2])
    
    # Try to find a solution for increasing M
    for M in range(105, M_max + 1):
        if time.time() > deadline - 5:
            break
            
        # Filter candidates
        candidates = [r for r in rows_data if r[2] <= M]
        
        if len(candidates) < n_rows:
            continue
            
        # Try to find n_rows disjoint rows
        solution = _find_disjoint_subset(candidates, n_rows, deadline)
        
        if solution:
            # Convert back to list of lists
            result = [list(r[0]) for r in solution]
            return result
            
    # If no solution found with M_max, return the best greedy or randomized result
    # Fallback to greedy to ensure valid output
    return _greedy_construct(n_rows, k)

def _generate_sidon_rows(M_max, k):
    """
    Generates all rows of length k+1 starting with 0, strictly increasing,
    max value <= M_max, such that all pairwise differences are distinct.
    Returns list of (row_tuple, diff_mask, max_val).
    """
    results = []
    
    def backtrack(current_row, current_diffs_mask, last_val):
        if len(current_row) == k + 1:
            max_val = current_row[-1]
            results.append((tuple(current_row), current_diffs_mask, max_val))
            return
            
        remaining = k + 1 - len(current_row)
        if M_max - last_val < remaining:
            return
            
        for next_val in range(last_val + 1, M_max + 1):
            new_diffs_mask = 0
            valid = True
            for prev_val in current_row:
                diff = next_val - prev_val
                bit = 1 << diff
                if current_diffs_mask & bit:
                    valid = False
                    break
                new_diffs_mask |= bit
            
            if not valid:
                continue
                
            current_row.append(next_val)
            backtrack(current_row, current_diffs_mask | new_diffs_mask, next_val)
            current_row.pop()
            
    backtrack([0], 0, 0)
    return results

def _find_disjoint_subset(candidates, n_rows, deadline):
    """
    Finds n_rows rows from candidates such that their diff masks are disjoint.
    Returns list of row tuples or None.
    """
    def backtrack(index, used_mask, selected):
        if time.time() > deadline:
            return None
            
        if len(selected) == n_rows:
            return selected[:]
            
        if index == len(candidates):
            return None
            
        if len(candidates) - index < n_rows - len(selected):
            return None
            
        row, mask, max_val = candidates[index]
        if (mask & used_mask) == 0:
            res = backtrack(index + 1, used_mask | mask, selected + [row])
            if res:
                return res
                
        res = backtrack(index + 1, used_mask, selected)
        if res:
            return res
            
        return None
        
    return backtrack(0, 0, [])

def _greedy_construct(n_rows, k):
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
