import random
import time

def construct():
    n = 10
    L1 = [[(i+j)%n for j in range(n)] for i in range(n)]
    
    def find_orthogonal_square():
        deadline = time.time() + 55.0
        
        # We use a simple backtracking with randomization
        # To speed up, we can fix the first row to 0..9 (since we can permute symbols in L2? No, L2 symbols are fixed 0..9, but we can permute the rows/cols of L2? No, L1 is fixed. We can permute the symbols of L2? Yes, any permutation of symbols in L2 preserves Latin property and orthogonality? No, orthogonality is preserved if we apply the same permutation to all symbols in L2. So we can fix L2[0][0]=0, L2[0][1]=1, etc?
        # Actually, we can fix the first row of L2 to be 0,1,...,9 without loss of generality?
        # If we permute the symbols of L2, the pairs (L1[i][j], L2[i][j]) change.
        # Wait, if we apply a permutation $\pi$ to the symbols of L2, the new pairs are $(L1[i][j], \pi(L2[i][j]))$.
        # The set of pairs is $\{ (a, \pi(b)) \mid (a,b) \in P \}$.
        # If $P$ contains all $(a,b)$, then the new set contains all $(a, \pi(b))$. Since $\pi$ is a bijection, the new set is also all distinct pairs.
        # So yes, we can fix the first row of L2 to be 0,1,...,9.
        
        grid = [[-1 for _ in range(n)] for _ in range(n)]
        # Fix first row
        for j in range(n):
            grid[0][j] = j
            # Update used sets
            # We need to track row, col, diag
            pass
        
        # This manual fixing complicates the backtracking setup.
        # Let's stick to the general backtracker, it's fast enough for n=10.
        
        def solve():
            grid = [[-1 for _ in range(n)] for _ in range(n)]
            row_used = [set() for _ in range(n)]
            col_used = [set() for _ in range(n)]
            diag_used = [set() for _ in range(n)]
            
            # Heuristic: Fill row by row.
            # We can also choose the next cell with minimum candidates.
            # For simplicity, row by row with random candidate shuffle.
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n * n:
                    return True
                
                i = idx // n
                j = idx % n
                k = (i + j) % n
                
                # Get candidates
                # Optimization: Pre-compute available values? No, sets are fast.
                candidates = [v for v in range(n) if v not in row_used[i] and v not in col_used[j] and v not in diag_used[k]]
                
                if not candidates:
                    return False
                
                random.shuffle(candidates)
                
                for v in candidates:
                    grid[i][j] = v
                    row_used[i].add(v)
                    col_used[j].add(v)
                    diag_used[k].add(v)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    grid[i][j] = -1
                    row_used[i].remove(v)
                    col_used[j].remove(v)
                    diag_used[k].remove(v)
                
                return False
            
            return backtrack(0)

        # Try a few times with different random seeds (implicitly via shuffle)
        for _ in range(1000):
            if time.time() > deadline:
                break
            if solve():
                # Re-construct grid from the last successful solve?
                # The `grid` inside `solve` is local. We need to return it.
                # Let's modify `solve` to return the grid.
                pass
        
        # Refined structure:
        
        def solve_with_grid():
            grid = [[-1 for _ in range(n)] for _ in range(n)]
            row_used = [set() for _ in range(n)]
            col_used = [set() for _ in range(n)]
            diag_used = [set() for _ in range(n)]
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n * n:
                    return True
                
                i = idx // n
                j = idx % n
                k = (i + j) % n
                
                candidates = [v for v in range(n) if v not in row_used[i] and v not in col_used[j] and v not in diag_used[k]]
                
                if not candidates:
                    return False
                
                random.shuffle(candidates)
                
                for v in candidates:
                    grid[i][j] = v
                    row_used[i].add(v)
                    col_used[j].add(v)
                    diag_used[k].add(v)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    grid[i][j] = -1
                    row_used[i].remove(v)
                    col_used[j].remove(v)
                    diag_used[k].remove(v)
                
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
