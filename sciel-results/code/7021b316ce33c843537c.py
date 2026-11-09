import sys
sys.setrecursionlimit(10000)

def construct():
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # L2[i][j] to be determined
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Precompute diagonals for L1
    # diag[a] = list of (i, j) such that (i+j)%n == a
    diags = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            a = (i + j) % n
            diags[a].append((i, j))
            
    # State for backtracking
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    diag_used = [set() for _ in range(n)] # diag_used[a] contains symbols used in L2 for cells in diag a
    
    def solve(idx):
        if idx == n * n:
            return True
            
        # Find next empty cell
        # We can iterate in row-major order, or pick the most constrained cell.
        # For n=10, row-major is likely fast enough.
        i = idx // n
        j = idx % n
        
        if L2[i][j] != -1:
            return solve(idx + 1)
            
        a = (i + j) % n
        
        # Try all symbols 0..9
        for sym in range(n):
            if sym in row_used[i]:
                continue
            if sym in col_used[j]:
                continue
            if sym in diag_used[a]:
                continue
                
            # Place sym
            L2[i][j] = sym
            row_used[i].add(sym)
            col_used[j].add(sym)
            diag_used[a].add(sym)
            
            if solve(idx + 1):
                return True
                
            # Backtrack
            L2[i][j] = -1
            row_used[i].remove(sym)
            col_used[j].remove(sym)
            diag_used[a].remove(sym)
            
        return False

    if solve(0):
        # Convert to pure Python ints and lists
        L1_clean = [[int(x) for x in row] for row in L1]
        L2_clean = [[int(x) for x in row] for row in L2]
        return [L1_clean, L2_clean]
    else:
        # Fallback to 1 square if solver fails (should not happen)
        L1_clean = [[int(x) for x in row] for row in L1]
        return [L1_clean]
