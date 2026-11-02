import numpy as np
import time
import random

def verify(U, V, W, n, m, p):
    """
    Verify if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Indexing:
    U[i*m + j, c]
    V[j*p + k, c]
    W[k*n + i, c]
    Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta(j, k)
    """
    rank = len(U[0]) if U else 0
    if not U or not V or not W:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1 if j == k else 0
                # Check if s is close to expected
                if abs(s - expected) > 1e-9:
                    return False
    return True

def try_find_decomposition(n, m, p, r, deadline):
    """
    Try to find a decomposition of rank r.
    Returns (U, V, W) as lists of lists, or None.
    """
    # Random search for U, V
    # Use small integers
    num_attempts = 100
    for attempt in range(num_attempts):
        if time.time() > deadline:
            return None
            
        # Generate random U, V with 0/1
        # Density can be tuned
        density = 0.3
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        
        for i in range(n * m):
            for c in range(r):
                if random.random() < density:
                    U[i][c] = 1
        for j in range(m * p):
            for c in range(r):
                if random.random() < density:
                    V[j][c] = 1
                    
        # Construct linear system A x = b
        # Variables x are W[k*n+i, c]
        # Num vars = p * n * r
        # Num eqs = n * m * p
        num_vars = p * n * r
        num_eqs = n * m * p
        
        A = np.zeros((num_eqs, num_vars))
        b = np.zeros(num_eqs)
        
        # Fill A and b
        # Row index mapping: (i, j, k) -> i*m*p + j*p + k
        # Col index mapping: (k, i, c) -> k*n*r + i*r + c
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_idx = i * m * p + j * p + k
                    b[row_idx] = 1 if j == k else 0
                    for c in range(r):
                        col_idx = k * n * r + i * r + c
                        A[row_idx, col_idx] = U[i * m + j][c] * V[j * p + k][c]
                        
        # Solve A x = b
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
            
        # Check if solution is close to integer
        # x is flattened W
        # We want x to be integer (or half-integer)
        # Let's check if x is close to round(x)
        x_rounded = np.round(x)
        if np.allclose(x, x_rounded, atol=1e-6):
            # Convert to lists
            W = [[0] * r for _ in range(p * n)]
            for k in range(p):
                for i in range(n):
                    for c in range(r):
                        W[k * n + i][c] = int(x_rounded[k * n * r + i * r + c])
            
            # Verify
            if verify(U, V, W, n, m, p):
                return U, V, W
        
        # If not integer, try to find a better solution?
        # For now, just restart
        
    return None

def construct():
    n, m, p = 3, 5, 7
    start_time = time.time()
    deadline = start_time + 120
    
    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, p*n) = 35
    # Let's try 36 to 60
    for r in range(36, 60):
        if time.time() > deadline - 10:
            break
            
        result = try_find_decomposition(n, m, p, r, deadline)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
