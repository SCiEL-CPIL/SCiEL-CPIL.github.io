import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    # 115s deadline to be safe
    deadline = start_time + 115.0
    
    # 1. Greedy Construction
    def greedy():
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

    best_rows = greedy()
    best_M = max(max(r) for r in best_rows)
    
    # 2. Backtracking Search
    # Total differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Theoretical lower bound for M
    # We need total_diffs distinct positive integers <= M
    # So M >= total_diffs
    lower_M = total_diffs
    
    # If greedy already hits the lower bound, we are done
    if best_M == lower_M:
        return best_rows

    # We search for M in [lower_M, best_M - 1]
    # We want the smallest M.
    
    # Helper for DFS
    def dfs_search(max_val):
        used_diffs = set()
        rows = []
        found = [None] # Use list for closure mutable
        
        def dfs(row_idx, pos, current_row):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                found[0] = [r[:] for r in rows]
                return True
            
            if pos == k + 1:
                last_val = current_row[-1]
                if last_val > max_val:
                    return False
                # Enforce non-decreasing max values for rows
                if row_idx > 0 and last_val < rows[row_idx - 1][-1]:
                    return False
                
                rows.append(current_row[:])
                if dfs(row_idx + 1, 1, [0]):
                    return True
                rows.pop()
                return False
            
            # Determine range for next value
            if pos == 0:
                x_start = 0
            else:
                x_start = current_row[-1] + 1
            
            x_end = max_val
            
            # Pruning: 
            # We need to place (k+1 - pos) more elements in this row.
            # The last element will be at least x_start + (k - pos) (if we pick consecutive integers)
            # Actually, min last element = current_row[-1] + (k+1 - pos)
            # If this min > max_val, prune.
            if current_row[-1] + (k + 1 - pos) > max_val:
                return False
            
            for x in range(x_start, x_end + 1):
                new_diffs = set()
                valid = True
                for i in range(pos):
                    d = x - current_row[i]
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                used_diffs.update(new_diffs)
                current_row.append(x)
                
                if dfs(row_idx, pos + 1, current_row):
                    return True
                
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        if dfs(0, 0, [0]):
            return found[0]
        return None

    # Try to find better M
    # We iterate from lower_M to best_M - 1
    # To avoid spending all time on the first (hardest) M, we can limit time per M?
    # Or just let the global deadline handle it.
    # Since lower_M is the absolute minimum, if it fails, it might be hard.
    # But the search space is smaller for smaller M.
    
    for M in range(lower_M, best_M):
        if time.time() > deadline - 5.0:
            break
        
        sol = dfs_search(M)
        if sol is not None:
            best_rows = sol
            best_M = M
            # If we found a solution at the lower bound, we can't do better
            if M == lower_M:
                break
            # If we found a solution, we could try to see if M-1 works?
            # No, we are iterating UP. So M is the first one that worked.
            # So it is the minimum.
            break

    return best_rows
