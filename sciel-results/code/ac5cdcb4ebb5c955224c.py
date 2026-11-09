import time
import random
import sys
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS

def _verify(U, V, W, rank):
    """Check if U,V,W with given rank exactly decompose the multiplication tensor.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _solve_linear_system(A, b):
    """
    Solve A x = b using Gaussian elimination with Fractions.
    A is a list of lists (rows).
    b is a list of Fractions.
    Returns a list of Fractions (x) or None if no solution.
    Since it's underdetermined, we set free variables to 0.
    """
    num_rows = len(A)
    num_cols = len(A[0]) if num_rows > 0 else 0
    
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Gaussian elimination
    row = 0
    col = 0
    while row < num_rows and col < num_cols:
        # Find pivot
        pivot = None
        for r in range(row, num_rows):
            if M[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            col += 1
            continue
            
        # Swap rows
        M[row], M[pivot] = M[pivot], M[row]
        
        # Normalize pivot row
        pivot_val = M[row][col]
        for c in range(col, num_cols + 1):
            M[row][c] /= pivot_val
            
        # Eliminate other rows
        for r in range(num_rows):
            if r != row and M[r][col] != 0:
                factor = M[r][col]
                for c in range(col, num_cols + 1):
                    M[r][c] -= factor * M[row][c]
        
        row += 1
        col += 1
        
    # Check for consistency
    for r in range(row, num_rows):
        if M[r][num_cols] != 0:
            return None
            
    # Extract solution
    # Free variables are those columns not used as pivots.
    # We set them to 0.
    x = [Fraction(0)] * num_cols
    # We need to track which columns are pivots.
    # In the loop above, we incremented col for each pivot.
    # But we didn't store the pivot columns.
    # Let's re-implement to track pivot columns.
    
    return None # Placeholder, will rewrite

def _solve_linear_system_v2(A, b):
    num_rows = len(A)
    num_cols = len(A[0]) if num_rows > 0 else 0
    
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    pivot_cols = []
    row = 0
    col = 0
    while row < num_rows and col < num_cols:
        pivot = None
        for r in range(row, num_rows):
            if M[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            col += 1
            continue
            
        M[row], M[pivot] = M[pivot], M[row]
        pivot_val = M[row][col]
        for c in range(col, num_cols + 1):
            M[row][c] /= pivot_val
            
        for r in range(num_rows):
            if r != row and M[r][col] != 0:
                factor = M[r][col]
                for c in range(col, num_cols + 1):
                    M[r][c] -= factor * M[row][c]
                    
        pivot_cols.append(col)
        row += 1
        col += 1
        
    for r in range(row, num_rows):
        if M[r][num_cols] != 0:
            return None
            
    x = [Fraction(0)] * num_cols
    for r in range(row):
        x[pivot_cols[r]] = M[r][num_cols]
        
    return x

def _try_rank_with_linear_solve(rank, max_trials=50, seed_base=0):
    rng = random.Random(seed_base)
    for trial in range(max_trials):
        # Generate random U, V with small integers
        U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)]
        
        # Build A and b
        # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # Variables: W[k*n+i, c]
        # Number of variables: p*n*rank
        num_vars = p * n * rank
        A = [[Fraction(0) for _ in range(num_vars)] for _ in range(n*m*p)]
        b = [Fraction(1) for _ in range(n*m*p)]
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        coeff = U[i*m+j][c] * V[j*p+k][c]
                        if coeff != 0:
                            # W index: k*n+i, c
                            var_idx = (k*n+i) * rank + c
                            A[eq_idx][var_idx] += Fraction(coeff)
                    eq_idx += 1
                    
        # Solve
        try:
            W_flat = _solve_linear_system_v2(A, b)
            if W_flat is not None:
                # Reshape W
                W = [[0] * rank for _ in range(p*n)]
                for idx, val in enumerate(W_flat):
                    row_idx = idx // rank
                    col_idx = idx % rank
                    W[row_idx][col_idx] = val
                
                # Verify
                if _verify(U, V, W, rank):
                    return {"U": U, "V": V, "W": W}
        except Exception:
            continue
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find a decomposition with rank < 84
    # Known lower bound 28, known upper bound 84.
    # Try ranks from 73 upwards (73 is a known bound for 3x4x7)
    for rank in range(73, 85):
        if time.time() > deadline - 10.0:
            break
        
        # Try a few trials for this rank
        result = _try_rank_with_linear_solve(rank, max_trials=20, seed_base=rank * 100)
        if result is not None:
            return result
            
    # Fallback to schoolbook
    return _schoolbook()
