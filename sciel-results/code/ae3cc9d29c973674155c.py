"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random
import numpy as np
from itertools import product

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Leave 10s safety margin

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _check_decomposition(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _check_decomposition_float(n, m, p, U, V, W, tol=1e-9):
    """Fast float check for approximate verification."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(n, m, p, U, V, W, max_iter=500):
    """Try to reduce rank by finding linear dependencies and eliminating columns."""
    rank = len(U[0])
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    
    for iteration in range(max_iter):
        if _time_left() < 5:
            break
        
        found = False
        for a in range(rank):
            if _time_left() < 5:
                break
            for b in range(a + 1, rank):
                same = True
                for r in range(len(U)):
                    if U[r][a] != U[r][b]:
                        same = False
                        break
                if same:
                    for r in range(len(V)):
                        if V[r][a] != V[r][b]:
                            same = False
                            break
                if same:
                    for r in range(len(W)):
                        if W[r][a] != W[r][b]:
                            same = False
                            break
                if same:
                    for row in U:
                        del row[b]
                    for row in V:
                        del row[b]
                    for row in W:
                        del row[b]
                    rank -= 1
                    found = True
                    break
            if found:
                break
        
        if not found:
            break
    
    return U, V, W

def _gf2_solve_linear_system(A, b):
    """Solve A*x = b over GF(2). A is list of lists, b is list of ints.
    Returns list of ints or None if no solution."""
    n_cols = len(A[0]) if A else 0
    n_rows = len(A)
    
    # Augmented matrix
    aug = [[int(A[i][j]) & 1 for j in range(n_cols)] + [int(b[i]) & 1] for i in range(n_rows)]
    
    # Gaussian elimination
    rank = 0
    for col in range(n_cols):
        # Find pivot
        pivot = -1
        for row in range(rank, n_rows):
            if aug[row][col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        
        # Swap
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        
        # Eliminate
        for row in range(n_rows):
            if row != rank and aug[row][col] == 1:
                for j in range(col, n_cols + 1):
                    aug[row][j] ^= aug[rank][j]
        
        rank += 1
    
    # Check for inconsistency
    for row in range(rank, n_rows):
        if any(aug[row][j] == 1 for j in range(n_cols)):
            return None
    
    # Back substitution
    x = [0] * n_cols
    for row in range(rank - 1, -1, -1):
        # Find the column with pivot in this row
        pivot_col = -1
        for col in range(n_cols):
            if aug[row][col] == 1:
                pivot_col = col
                break
        if pivot_col == -1:
            continue
        val = aug[row][n_cols]
        for col in range(pivot_col + 1, n_cols):
            if aug[row][col] == 1:
                val ^= x[col]
        x[pivot_col] = val
    
    return x

def _try_gf2_search(n, m, p, max_rank=25, max_attempts=200):
    """Search for low-rank decomposition over GF(2), then lift to integers."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    for target_rank in range(20, max_rank + 1):
        if _time_left() < 10:
            break
        
        for attempt in range(max_attempts):
            if _time_left() < 5:
                break
            
            # Random initialization over GF(2)
            U = [[random.randint(0, 1) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.randint(0, 1) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.randint(0, 1) for _ in range(target_rank)] for _ in range(pn)]
            
            # Check if this works over GF(2)
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s = (s + U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]) & 1
                        if s != 1:
                            valid = False
                            break
            
            if valid:
                # Lift to integers (0 stays 0, 1 stays 1)
                return U, V, W, target_rank
    
    return None, None, None, None

def _structured_search(n, m, p):
    """Try structured ansatz based on known 3x3x3 rank-20 decomposition."""
    # For 3x3x3, there's a known rank-20 decomposition
    # We can try to extend it to 3x3x4
    
    # One approach: use the 3x3x3 rank-20 and add 3 more terms for the 4th column
    # But this might not be optimal
    
    # Alternative: try random search with structured initialization
    nm = n * m
    mp = m * p
    pn = p * n
    
    best_rank = nm * mp * pn  # schoolbook
    best_U = [[0] * best_rank for _ in range(nm)]
    best_V = [[0] * best_rank for _ in range(mp)]
    best_W = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Try target ranks from 20 upward
    for target_rank in range(20, min(30, best_rank + 1)):
        if _time_left() < 10:
            break
        
        for attempt in range(100):
            if _time_left() < 5:
                break
            
            # Random initialization with small integers
            U_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(nm)]
            V_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(mp)]
            W_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(pn)]
            
            # Quick float check
            if _check_decomposition_float(n, m, p, U_rand, V_rand, W_rand, tol=1e-6):
                if _check_decomposition(n, m, p, U_rand, V_rand, W_rand):
                    return U_rand, V_rand, W_rand, target_rank
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Start with schoolbook
    rank = nm * mp * pn
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    assert _check_decomposition(n, m, p, U, V, W), "Schoolbook decomposition failed"
    
    # Try to reduce rank via simple column elimination
    U, V, W = _try_reduce_rank(n, m, p, U, V, W, max_iter=100)
    current_rank = len(U[0])
    
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    # Try GF(2) search for rank 20-25
    gf2_result = _try_gf2_search(n, m, p, max_rank=25, max_attempts=300)
    if gf2_result[0] is not None:
        gf2_U, gf2_V, gf2_W, gf2_rank = gf2_result
        if gf2_rank < best_rank:
            best_U, best_V, best_W = gf2_U, gf2_V, gf2_W
            best_rank = gf2_rank
    
    # Try structured search
    struct_result = _structured_search(n, m, p)
    if struct_result[3] < best_rank:
        best_U, best_V, best_W = struct_result[0], struct_result[1], struct_result[2]
        best_rank = struct_result[3]
    
    # Final random search for lower ranks
    for target_rank in range(max(18, best_rank - 5), best_rank):
        if _time_left() < 10:
            break
        
        for attempt in range(200):
            if _time_left() < 5:
                break
            
            U_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(nm)]
            V_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(mp)]
            W_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(pn)]
            
            if _check_decomposition_float(n, m, p, U_rand, V_rand, W_rand, tol=1e-6):
                if _check_decomposition(n, m, p, U_rand, V_rand, W_rand):
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    best_rank = target_rank
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
