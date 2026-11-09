"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Leave 15s safety margin

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

def _strassen_2x2x2(U, V, W, start_col, i_off, j_off, k_off, n, m, p):
    """Add a 2x2x2 Strassen block starting at start_col.
    i_off, j_off: offset in A (i,j)
    j_off, k_off: offset in B (j,k)
    k_off, i_off: offset in C (k,i)
    Returns number of columns used (7).
    """
    col = start_col
    # Strassen 7 multiplications for 2x2x2
    # A is 2x2, B is 2x2, C = A*B
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U entries: A factors
    # V entries: B factors
    # W entries: C coefficients
    
    # M1
    U[(i_off+0)*m + (j_off+0)][col] = 1
    U[(i_off+1)*m + (j_off+1)][col] = 1
    V[(j_off+0)*p + (k_off+0)][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = 1
    W[(k_off+0)*n + (i_off+0)][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M2
    U[(i_off+1)*m + (j_off+0)][col] = 1
    U[(i_off+1)*m + (j_off+1)][col] = 1
    V[(j_off+0)*p + (k_off+0)][col] = 1
    W[(k_off+0)*n + (i_off+1)][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M3
    U[(i_off+0)*m + (j_off+0)][col] = 1
    V[(j_off+1)*p + (k_off+0)][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = -1
    W[(k_off+0)*n + (i_off+0)][col] = 1
    W[(k_off+1)*n + (i_off+0)][col] = 1
    col += 1
    
    # M4
    U[(i_off+1)*m + (j_off+1)][col] = 1
    V[(j_off+0)*p + (k_off+1)][col] = 1
    V[(j_off+0)*p + (k_off+0)][col] = -1
    W[(k_off+0)*n + (i_off+0)][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M5
    U[(i_off+0)*m + (j_off+0)][col] = 1
    U[(i_off+0)*m + (j_off+1)][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = 1
    W[(k_off+0)*n + (i_off+0)][col] = -1
    W[(k_off+1)*n + (i_off+0)][col] = 1
    col += 1
    
    # M6
    U[(i_off+1)*m + (j_off+0)][col] = 1
    U[(i_off+0)*m + (j_off+0)][col] = -1
    V[(j_off+0)*p + (k_off+0)][col] = 1
    V[(j_off+1)*p + (k_off+0)][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M7
    U[(i_off+0)*m + (j_off+1)][col] = 1
    U[(i_off+1)*m + (j_off+1)][col] = -1
    V[(j_off+0)*p + (k_off+1)][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = 1
    W[(k_off+0)*n + (i_off+0)][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = -1
    col += 1
    
    return 7

def _simple_block(U, V, W, start_col, i_off, j_off, k_off, n, m, p, size):
    """Add a simple size x size x size block multiplication.
    Returns number of columns used (size^2).
    """
    col = start_col
    for ii in range(size):
        for jj in range(size):
            for kk in range(size):
                U[(i_off+ii)*m + (j_off+jj)][col] = 1
                V[(j_off+jj)*p + (k_off+kk)][col] = 1
                W[(k_off+kk)*n + (i_off+ii)][col] = 1
                col += 1
    return size * size

def _build_strassen_based():
    """Build a decomposition using Strassen for 2x2 blocks and simple for rest.
    
    A is 3x3, B is 3x4, C is 3x4.
    Split A into:
      A11 = A[0:2, 0:2] (2x2)
      A12 = A[0:2, 2:3] (2x1)
      A21 = A[2:3, 0:2] (1x2)
      A22 = A[2:3, 2:3] (1x1)
    
    Split B into:
      B11 = B[0:2, 0:2] (2x2)
      B12 = B[0:2, 2:4] (2x2)
      B21 = B[2:3, 0:2] (1x2)
      B22 = B[2:3, 2:4] (1x2)
    
    C = A*B:
      C11 = A11*B11 + A12*B21  (2x2)
      C12 = A11*B12 + A12*B22  (2x2)
      C21 = A21*B11 + A22*B21  (1x2)
      C22 = A21*B12 + A22*B22  (1x2)
    
    We can use Strassen for the 2x2x2 blocks where both A and B submatrices are 2x2:
      A11*B11: 2x2 x 2x2 -> Strassen (7)
      A11*B12: 2x2 x 2x2 -> Strassen (7)
    
    For mixed sizes, use simple multiplication:
      A12*B21: 2x1 x 1x2 -> 2 multiplications
      A12*B22: 2x1 x 1x2 -> 2 multiplications
      A21*B11: 1x2 x 2x2 -> 2 multiplications
      A21*B12: 1x2 x 2x2 -> 2 multiplications
      A22*B21: 1x1 x 1x2 -> 2 multiplications
      A22*B22: 1x1 x 1x2 -> 2 multiplications
    
    Total: 7 + 7 + 2 + 2 + 2 + 2 + 2 + 2 = 26
    """
    n, m, p = 3, 3, 4
    total_rank = 26
    U = [[0] * total_rank for _ in range(n*m)]
    V = [[0] * total_rank for _ in range(m*p)]
    W = [[0] * total_rank for _ in range(p*n)]
    
    col = 0
    # A11*B11: Strassen, i_off=0, j_off=0, k_off=0
    col += _strassen_2x2x2(U, V, W, col, 0, 0, 0, n, m, p)
    
    # A11*B12: Strassen, i_off=0, j_off=0, k_off=2
    col += _strassen_2x2x2(U, V, W, col, 0, 0, 2, n, m, p)
    
    # A12*B21: simple, i_off=0, j_off=2, k_off=0, size=1 but A is 2x1, B is 1x2
    # This is a 2x1 x 1x2 block. We need 2 multiplications.
    # A12 is rows 0,1 col 2. B21 is row 2 cols 0,1.
    # C contribution: C[0:2, 0:2] += A12 * B21
    # C[i,k] += A[i,2] * B[2,k] for i in 0,1 and k in 0,1
    for ii in range(2):
        for kk in range(2):
            U[(0+ii)*m + 2][col] = 1
            V[2*p + (0+kk)][col] = 1
            W[(0+kk)*n + (0+ii)][col] = 1
            col += 1
    
    # A12*B22: simple, i_off=0, j_off=2, k_off=2
    # A12 is rows 0,1 col 2. B22 is row 2 cols 2,3.
    # C[i,k] += A[i,2] * B[2,k] for i in 0,1 and k in 2,3
    for ii in range(2):
        for kk in range(2, 4):
            U[(0+ii)*m + 2][col] = 1
            V[2*p + kk][col] = 1
            W[kk*n + (0+ii)][col] = 1
            col += 1
    
    # A21*B11: simple, i_off=2, j_off=0, k_off=0
    # A21 is row 2 cols 0,1. B11 is rows 0,1 cols 0,1.
    # C[2,k] += A[2,j] * B[j,k] for j in 0,1 and k in 0,1
    for jj in range(2):
        for kk in range(2):
            U[2*m + jj][col] = 1
            V[jj*p + kk][col] = 1
            W[kk*n + 2][col] = 1
            col += 1
    
    # A21*B12: simple, i_off=2, j_off=0, k_off=2
    # A21 is row 2 cols 0,1. B12 is rows 0,1 cols 2,3.
    # C[2,k] += A[2,j] * B[j,k] for j in 0,1 and k in 2,3
    for jj in range(2):
        for kk in range(2, 4):
            U[2*m + jj][col] = 1
            V[jj*p + kk][col] = 1
            W[kk*n + 2][col] = 1
            col += 1
    
    # A22*B21: simple, i_off=2, j_off=2, k_off=0
    # A22 is row 2 col 2. B21 is row 2 cols 0,1.
    # C[2,k] += A[2,2] * B[2,k] for k in 0,1
    for kk in range(2):
        U[2*m + 2][col] = 1
        V[2*p + kk][col] = 1
        W[kk*n + 2][col] = 1
        col += 1
    
    # A22*B22: simple, i_off=2, j_off=2, k_off=2
    # A22 is row 2 col 2. B22 is row 2 cols 2,3.
    # C[2,k] += A[2,2] * B[2,k] for k in 2,3
    for kk in range(2, 4):
        U[2*m + 2][col] = 1
        V[2*p + kk][col] = 1
        W[kk*n + 2][col] = 1
        col += 1
    
    assert col == total_rank, f"Column count mismatch: {col} != {total_rank}"
    return U, V, W

def _try_reduce_rank(n, m, p, U, V, W, max_iter=200):
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
                # Check if columns are identical
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

def construct():
    n, m, p = DIMENSIONS
    
    # Start with Strassen-based rank-26 decomposition
    U, V, W = _build_strassen_based()
    
    # Verify
    assert _check_decomposition(n, m, p, U, V, W), "Strassen-based decomposition failed"
    
    # Try to reduce rank
    U, V, W = _try_reduce_rank(n, m, p, U, V, W, max_iter=100)
    
    current_rank = len(U[0])
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    # Try random search for lower rank
    for target_rank in range(best_rank - 1, max(15, best_rank - 10), -1):
        if _time_left() < 10:
            break
        
        for attempt in range(30):
            if _time_left() < 10:
                break
            
            U_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
            V_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
            W_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _check_decomposition_float(n, m, p, U_rand, V_rand, W_rand, tol=1e-6):
                if _check_decomposition(n, m, p, U_rand, V_rand, W_rand):
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    best_rank = target_rank
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
