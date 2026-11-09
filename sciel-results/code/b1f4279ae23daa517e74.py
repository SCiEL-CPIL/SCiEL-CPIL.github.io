import time
import numpy as np

_DEADLINE = 145.0

def _check_and_update(matrices, powers, n, u, v, c):
    """
    Checks if adding edge (u, v) with color c creates a C6.
    If not, updates the matrices and powers.
    Returns True if valid and updated, False otherwise.
    """
    A = matrices[c]
    # Check if path of length 5 exists between u and v in A
    # A^5[u, v] = sum_k A^4[u, k] * A[k, v]
    # We have A^4 stored in powers[c][4]
    A4 = powers[c][4]
    # A[k, v] is the column v of A
    # Since A is symmetric, A[:, v] is the same as A[v, :]
    # So A^5[u, v] = dot(A4[u, :], A[v, :])
    if np.dot(A4[u, :], A[v, :]) > 0:
        return False
    
    # Update A
    A[u, v] = 1
    A[v, u] = 1
    
    # Update powers
    # A2 = A @ A
    # A3 = A2 @ A
    # A4 = A2 @ A2
    
    # It is faster to recompute the small matrices for n<=20
    # n=17, 17x17 matrix mult is very fast with numpy
    A2 = A @ A
    A3 = A2 @ A
    A4 = A2 @ A2
    
    powers[c][2] = A2
    powers[c][3] = A3
    powers[c][4] = A4
    
    return True

def _undo_update(matrices, powers, n, u, v, c):
    """
    Undoes the addition of edge (u, v) with color c.
    """
    A = matrices[c]
    A[u, v] = 0
    A[v, u] = 0
    
    # Recompute powers
    A2 = A @ A
    A3 = A2 @ A
    A4 = A2 @ A2
    
    powers[c][2] = A2
    powers[c][3] = A3
    powers[c][4] = A4

def _solve_n(n, t_end):
    if n < 1:
        return None
        
    # Initialize matrices
    matrices = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
    powers = [[np.zeros((n, n), dtype=np.int32) for _ in range(5)] for _ in range(4)]
    
    # Precompute initial powers (all zero)
    # They are already zero.
    
    # Edge colors storage
    # We will store colors in a list of lists or a flat array
    # For backtracking, we need to assign colors to edges (i, j)
    # We iterate i from 0 to n-1, and for each i, j from i+1 to n-1
    # Actually, we add vertices one by one.
    # When adding vertex k, we assign colors to (k, 0), ..., (k, k-1)
    
    # Let's use a recursive backtracking
    # State: current vertex k being added
    # We need to store the colors assigned for previous vertices
    
    # To make it fast, we will use a list `colors` where colors[i][j] is the color of edge (i,j)
    # But we build incrementally.
    
    # We'll use a global-like structure in the closure or pass around
    # Let's define a nested function for backtracking
    
    import sys
    sys.setrecursionlimit(10000)
    
    # We need to track the current assignment
    # Since we are adding vertex k, the edges (k, 0)...(k, k-1) are being decided
    # The edges among 0...k-1 are already fixed.
    
    # We will store the colors in a 2D list `col`
    col = [[-1 for _ in range(n)] for _ in range(n)]
    
    def backtrack(k):
        if time.time() > t_end:
            return False
            
        if k == n:
            return True
            
        # We need to assign colors to edges (k, 0), ..., (k, k-1)
        # Let's do this one by one: j = 0, 1, ..., k-1
        
        def assign(j):
            if time.time() > t_end:
                return False
                
            if j == k:
                return backtrack(k + 1)
                
            for c in range(4):
                # Try color c for edge (k, j)
                # Note: edge (j, k) is the same as (k, j)
                if _check_and_update(matrices, powers, n, j, k, c):
                    col[k][j] = c
                    col[j][k] = c
                    
                    if assign(j + 1):
                        return True
                        
                    # Undo
                    col[k][j] = -1
                    col[j][k] = -1
                    _undo_update(matrices, powers, n, j, k, c)
                    
            return False
            
        return assign(0)
        
    if backtrack(0):
        # Extract colors in lexicographic order
        res = []
        for i in range(n):
            for j in range(i + 1, n):
                res.append(col[i][j])
        return res
        
    return None

def construct():
    t0 = time.time()
    t_end = t0 + _DEADLINE
    
    for n in [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > t_end - 1.0:
            break
            
        res = _solve_n(n, t_end)
        if res is not None:
            return {'n': n, 'colors': res}
            
    return {'n': 1, 'colors': []}
