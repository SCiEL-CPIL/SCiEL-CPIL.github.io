"""Matrix multiplication 2x3x4 rank minimization via structured search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W, n=2, m=3, p=4):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0
                rhs = 0.0
                for c in range(rank):
                    rhs += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(lhs - rhs) > 1e-9:
                    return False
    return True

def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < tol

def _verify_half_integer(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for M in [U, V, W]:
        for row in M:
            for val in row:
                if not _is_half_integer(val, tol):
                    return False
    return True

def _try_rank(n, m, p, rank, deadline, seed=0):
    """Try to find a decomposition of the given rank."""
    random.seed(seed)
    start = time.time()
    
    # Value sets to try, from small to larger
    value_sets = [
        [-1, 0, 1],
        [-1, -0.5, 0, 0.5, 1],
        [-1, -0.5, 0, 0.5, 1, 1.5, 2],
    ]
    
    for values in value_sets:
        while time.time() < deadline - 5:
            # Random U and V
            U = [[random.choice(values) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice(values) for _ in range(rank)] for _ in range(m*p)]
            
            # Build linear system for W
            W = [[0]*rank for _ in range(p*n)]
            valid = True
            
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    b = np.ones(m)
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U[i*m+j][c] * V[j*p+k][c]
                    
                    # Check if system is solvable
                    if np.linalg.matrix_rank(A) < m:
                        valid = False
                        break
                    
                    # Solve least squares
                    W_row, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    if np.max(np.abs(A @ W_row - b)) > 1e-9:
                        valid = False
                        break
                    if not all(abs(W_row[c] * 2 - round(W_row[c] * 2)) < 1e-9 for c in range(rank)):
                        valid = False
                        break
                    W[k*n+i] = [float(x) for x in W_row]
                
                if not valid:
                    break
            
            if valid:
                if _verify(U, V, W, n, m, p) and _verify_half_integer(U, V, W):
                    return (U, V, W)
    
    return None

def _structured_search(n, m, p, rank, deadline):
    """Try structured patterns for the decomposition."""
    start = time.time()
    
    # Try sparse patterns with specific structures
    patterns = []
    
    # Pattern 1: Block diagonal structure
    # Split the rank into blocks corresponding to different (i,k) pairs
    for i in range(n):
        for k in range(p):
            # Each (i,k) pair needs m=3 terms
            # But we can share terms across i and k
            pass
    
    # Try random sparse patterns with more structure
    for attempt in range(1000):
        if time.time() > deadline - 5:
            break
        
        # Create sparse U and V with specific patterns
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        
        # Assign non-zero entries in a structured way
        for c in range(rank):
            # Each column c contributes to a specific pattern
            # Randomly choose which (i,j) and (j,k) pairs are non-zero
            i_idx = random.randint(0, n-1)
            j_idx = random.randint(0, m-1)
            k_idx = random.randint(0, p-1)
            
            U[i_idx*m+j_idx][c] = random.choice([-1, 0.5, 1])
            V[j_idx*p+k_idx][c] = random.choice([-1, 0.5, 1])
        
        # Now solve for W
        W = [[0]*rank for _ in range(p*n)]
        valid = True
        
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i*m+j][c] * V[j*p+k][c]
                
                if np.linalg.matrix_rank(A) < m:
                    valid = False
                    break
                
                W_row, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                if np.max(np.abs(A @ W_row - b)) > 1e-9:
                    valid = False
                    break
                if not all(abs(W_row[c] * 2 - round(W_row[c] * 2)) < 1e-9 for c in range(rank)):
                    valid = False
                    break
                W[k*n+i] = [float(x) for x in W_row]
            
            if not valid:
                break
        
        if valid:
            if _verify(U, V, W, n, m, p) and _verify_half_integer(U, V, W):
                return (U, V, W)
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x4 matrix multiplication."""
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24
    
    # Start with schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_solution = (U_sb, V_sb, W_sb)
    
    # Try ranks from 12 upward (theoretical minimum is 12)
    for target_rank in range(12, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try random search
        found = _try_rank(n, m, p, target_rank, deadline - 5, seed=target_rank*1000)
        if found is not None:
            U, V, W = found
            if _verify(U, V, W, n, m, p) and _verify_half_integer(U, V, W):
                best_rank = target_rank
                best_solution = (U, V, W)
                break
        
        # Try structured search
        if time.time() < deadline - 10:
            found = _structured_search(n, m, p, target_rank, deadline - 5)
            if found is not None:
                U, V, W = found
                if _verify(U, V, W, n, m, p) and _verify_half_integer(U, V, W):
                    best_rank = target_rank
                    best_solution = (U, V, W)
                    break
    
    U, V, W = best_solution
    return {"U": U, "V": V, "W": W}
