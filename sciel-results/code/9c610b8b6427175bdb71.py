"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _validate_numpy(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Vectorized validation using numpy for speed."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
                val = np.sum(prod)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _clean_entry(v):
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    doubled = v * 2
    if abs(doubled - round(doubled)) < 1e-9:
        return round(doubled) / 2.0
    return v

def _construct_rank_36():
    """
    Rank-36 decomposition for 2x4 x 4x5 matrix multiplication.
    
    Strategy: Split into two 2x2x5 sub-problems.
    Each 2x2x5 splits into two 2x2x2 (Strassen, rank 7) + one 2x2x1 (schoolbook, rank 4).
    Total rank: 2 * (7 + 7 + 4) = 36.
    
    U: shape (n*m, rank) = (8, 36)
    V: shape (m*p, rank) = (20, 36)  
    W: shape (p*n, rank) = (10, 36)
    
    Indexing:
    U[i*m+j, c] where i in [0,2), j in [0,4)
    V[j*p+k, c] where j in [0,4), k in [0,5)
    W[k*n+i, c] where k in [0,5), i in [0,2)
    """
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        """Add Strassen 2x2x2 block for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1] = C[i1:i2+1, k1:k2+1]"""
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (A11+A22)(B11+B22)
        col = col_offset + 0
        U[a11][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        V[b22][col] = 1
        W[c11][col] = 1
        W[c22][col] = 1
        
        # M2 = (A21+A22)B11
        col = col_offset + 1
        U[a21][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1
        W[c22][col] = -1
        
        # M3 = A11(B12-B22)
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1
        V[b22][col] = -1
        W[c12][col] = 1
        W[c22][col] = 1
        
        # M4 = A22(B21-B11)
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1
        V[b11][col] = -1
        W[c11][col] = 1
        W[c21][col] = 1
        
        # M5 = (A11+A12)B22
        col = col_offset + 4
        U[a11][col] = 1
        U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1
        W[c12][col] = 1
        
        # M6 = (A21-A11)(B11+B12)
        col = col_offset + 5
        U[a21][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b12][col] = 1
        W[c22][col] = 1
        
        # M7 = (A12-A22)(B21+B22)
        col = col_offset + 6
        U[a12][col] = 1
        U[a22][col] = -1
        V[b21][col] = 1
        V[b22][col] = 1
        W[c11][col] = 1

    def add_schoolbook_221(i1, i2, j1, j2, k1, col_offset):
        """Add schoolbook 2x2x1 block for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1] = C[i1:i2+1, k1]"""
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b21 = j2 * p + k1
        
        c11 = k1 * n + i1
        c21 = k1 * n + i2
        
        # Term 1: A11 * B11 -> contributes to C11
        col = col_offset + 0
        U[a11][col] = 1
        V[b11][col] = 1
        W[c11][col] = 1
        
        # Term 2: A12 * B21 -> contributes to C11
        col = col_offset + 1
        U[a12][col] = 1
        V[b21][col] = 1
        W[c11][col] = 1
        
        # Term 3: A21 * B11 -> contributes to C21
        col = col_offset + 2
        U[a21][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1
        
        # Term 4: A22 * B21 -> contributes to C21
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1
        W[c21][col] = 1

    col = 0
    
    # Sub-problem 1: A rows 0-1, cols 0-1; B rows 0-1, cols 0-4
    # 2x2x2 for cols 0,1
    add_strassen_222(0, 1, 0, 1, 0, 1, col)
    col += 7
    # 2x2x2 for cols 2,3
    add_strassen_222(0, 1, 0, 1, 2, 3, col)
    col += 7
    # 2x2x1 for col 4
    add_schoolbook_221(0, 1, 0, 1, 4, col)
    col += 4
    
    # Sub-problem 2: A rows 0-1, cols 2-3; B rows 2-3, cols 0-4
    # 2x2x2 for cols 0,1
    add_strassen_222(0, 1, 2, 3, 0, 1, col)
    col += 7
    # 2x2x2 for cols 2,3
    add_strassen_222(0, 1, 2, 3, 2, 3, col)
    col += 7
    # 2x2x1 for col 4
    add_schoolbook_221(0, 1, 2, 3, 4, col)
    col += 4
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return U, V, W

def _construct_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 40."""
    n, m, p = 2, 4, 5
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
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try Rank-36 decomposition (Strassen-based)
    U_rank36, V_rank36, W_rank36 = _construct_rank_36()
    if _validate_numpy(U_rank36, V_rank36, W_rank36, n, m, p):
        result_U = [[_clean_entry(v) for v in row] for row in U_rank36]
        result_V = [[_clean_entry(v) for v in row] for row in V_rank36]
        result_W = [[_clean_entry(v) for v in row] for row in W_rank36]
        return {"U": result_U, "V": result_V, "W": result_W}
    
    # Fallback to Schoolbook
    U_school, V_school, W_school = _construct_schoolbook()
    assert _validate_numpy(U_school, V_school, W_school, n, m, p), "Schoolbook decomposition failed!"
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
