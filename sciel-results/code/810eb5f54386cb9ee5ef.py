import time
import random

def construct():
    n = 10
    L1 = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
        [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
        [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
        [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
        [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
        [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
        [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
        [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    ]
    
    # Check if we can find a second square
    # Use a deadline
    deadline = time.time() + 45.0
    
    # Backtracking solver for L2 orthogonal to L1
    # L2[i][j]
    # Constraints:
    # 1. Rows are permutations
    # 2. Cols are permutations
    # 3. For each k in 0..9, {L2[i][j] | L1[i][j]==k} is a permutation
    
    # Precompute the indices for each L1 value
    l1_indices = {k: [] for k in range(n)}
    for i in range(n):
        for j in range(n):
            l1_indices[L1[i][j]].append((i, j))
            
    # We can fix L2[0][j] = j to break symmetry?
    # Let's try without fixing first, or fix to speed up.
    # Fixing L2[0] = [0..9] is safe due to symbol relabeling of L2.
    
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    # Fix first row
    for j in range(n):
        L2[0][j] = j
        
    # Used in rows, cols, l1-classes
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    l1_used = [set() for _ in range(n)]
    
    # Initialize with first row
    for j in range(n):
        row_used[0].add(j)
        col_used[j].add(j)
        l1_used[L1[0][j]].add(j)
        
    # Cells to fill
    cells = [(i, j) for i in range(1, n) for j in range(n)]
    
    # Shuffle cells to avoid deterministic bias?
    # Better to use MRV (Minimum Remaining Values)
    
    def get_domains(i, j):
        used = row_used[i] | col_used[j] | l1_used[L1[i][j]]
        return [v for v in range(n) if v not in used]
        
    def backtrack(idx):
        if time.time() > deadline:
            return False
            
        if idx == len(cells):
            return True
            
        # MRV: Find the cell with fewest options
        min_idx = idx
        min_count = float('inf')
        
        # Check from current idx to end (or all unassigned?)
        # For simplicity and speed, just pick the first one with fewest options in the remaining list
        # But checking all remaining is O(N^2) per step.
        # Let's just iterate and pick the best.
        
        best_i, best_j = -1, -1
        best_count = float('inf')
        
        for k in range(idx, len(cells)):
            i, j = cells[k]
            if L2[i][j] != -1: continue
            doms = get_domains(i, j)
            if len(doms) == 0:
                return False
            if len(doms) < best_count:
                best_count = len(doms)
                best_i, best_j = i, j
                min_idx = k
                if best_count == 1:
                    break
        
        if best_i == -1:
            return True # All filled? (Should be caught by idx==len)
            
        # Swap min_idx with idx to process it now
        cells[idx], cells[min_idx] = cells[min_idx], cells[idx]
        
        i, j = cells[idx]
        doms = get_domains(i, j)
        
        for v in doms:
            L2[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            l1_used[L1[i][j]].add(v)
            
            if backtrack(idx + 1):
                return True
                
            L2[i][j] = -1
            row_used[i].remove(v)
            col_used[j].remove(v)
            l1_used[L1[i][j]].remove(v)
            
        return False

    if backtrack(0):
        # Verify L2 is valid
        valid = True
        # Check rows
        for r in range(n):
            if sorted(L2[r]) != list(range(n)):
                valid = False
                break
        # Check cols
        if valid:
            for c in range(n):
                col = [L2[r][c] for r in range(n)]
                if sorted(col) != list(range(n)):
                    valid = False
                    break
        # Check orthogonality
        if valid:
            pairs = set()
            for i in range(n):
                for j in range(n):
                    pairs.add((L1[i][j], L2[i][j]))
            if len(pairs) != n*n:
                valid = False
                
        if valid:
            return [L1, L2]
            
    return [L1]
