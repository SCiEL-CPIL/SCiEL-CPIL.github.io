import random
import time

def construct():
    n = 10
    L1 = [[(i+j)%n for j in range(n)] for i in range(n)]
    
    def find_orthogonal_square():
        deadline = time.time() + 55.0
        start_time = time.time()
        
        # Fix first row: L2[0][j] = j for all j
        # This is valid because symbol permutation preserves orthogonality
        
        def solve_with_grid():
            grid = [[-1 for _ in range(n)] for _ in range(n)]
            # Fix first row
            for j in range(n):
                grid[0][j] = j
            
            # Use bitmasks for row, col, diag constraints
            # row_used[i] is a bitmask of values used in row i
            # col_used[j] is a bitmask of values used in col j
            # diag_used[k] is a bitmask of values used in diagonal k=(i+j)%n
            
            row_used = [0] * n
            col_used = [0] * n
            diag_used = [0] * n
            
            # Initialize with first row
            for j in range(n):
                v = grid[0][j]
                row_used[0] |= (1 << v)
                col_used[j] |= (1 << v)
                k = (0 + j) % n
                diag_used[k] |= (1 << v)
            
            # Track which cells are unfilled
            unfilled = []
            for i in range(1, n):  # Skip row 0
                for j in range(n):
                    unfilled.append((i, j))
            
            def get_candidates(i, j):
                """Get available values for cell (i,j) as a list"""
                k = (i + j) % n
                mask = row_used[i] | col_used[j] | diag_used[k]
                candidates = []
                for v in range(n):
                    if not (mask & (1 << v)):
                        candidates.append(v)
                return candidates
            
            # MRV heuristic: pick cell with fewest candidates
            def find_mrv_cell():
                min_count = n + 1
                min_cell = None
                for (i, j) in unfilled:
                    k = (i + j) % n
                    mask = row_used[i] | col_used[j] | diag_used[k]
                    count = n - bin(mask).count('1')
                    if count < min_count:
                        min_count = count
                        min_cell = (i, j)
                    if min_count == 0:
                        break
                return min_cell, min_count
            
            def backtrack(depth):
                if time.time() > deadline:
                    return False
                
                if depth == len(unfilled):
                    return True
                
                # MRV: pick cell with fewest candidates
                cell, count = find_mrv_cell()
                if cell is None or count == 0:
                    return False
                
                i, j = cell
                k = (i + j) % n
                mask = row_used[i] | col_used[j] | diag_used[k]
                
                candidates = []
                for v in range(n):
                    if not (mask & (1 << v)):
                        candidates.append(v)
                
                random.shuffle(candidates)
                
                for v in candidates:
                    grid[i][j] = v
                    row_used[i] |= (1 << v)
                    col_used[j] |= (1 << v)
                    diag_used[k] |= (1 << v)
                    unfilled.remove(cell)
                    
                    if backtrack(depth + 1):
                        return True
                    
                    unfilled.append(cell)
                    grid[i][j] = -1
                    row_used[i] &= ~(1 << v)
                    col_used[j] &= ~(1 << v)
                    diag_used[k] &= ~(1 << v)
                
                return False
            
            if backtrack(0):
                return grid
            return None
        
        for _ in range(500):
            if time.time() > deadline:
                break
            res = solve_with_grid()
            if res:
                return res
        return None

    L2 = find_orthogonal_square()
    
    if L2:
        def is_latin(square):
            for i in range(n):
                if sorted(square[i]) != list(range(n)):
                    return False
                col = [square[j][i] for j in range(n)]
                if sorted(col) != list(range(n)):
                    return False
            return True

        def are_orthogonal(s1, s2):
            pairs = set()
            for i in range(n):
                for j in range(n):
                    pairs.add((s1[i][j], s2[i][j]))
            return len(pairs) == n*n

        if is_latin(L2) and are_orthogonal(L1, L2):
            L2_clean = [[int(L2[i][j]) for j in range(n)] for i in range(n)]
            return [L1, L2_clean]
            
    return [L1]
