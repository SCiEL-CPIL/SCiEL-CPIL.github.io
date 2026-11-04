"""Tensor decomposition for 4x5x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1   # C00 += M1
    W[3,0] = 1   # C11 += M1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1   # C10 += M2
    
    # M3 = A00(B10-B11)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1   # C01 += M3
    W[3,2] = -1  # C11 -= M3
    
    # M4 = A11(B01-B00)
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = -1  # C00 -= M4
    W[2,3] = 1   # C10 += M4
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = 1   # C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = 1   # C00 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[3,6] = 1   # C11 += M7
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    
    The formula is:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    """
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for c in range(rank):
        U_c = U[:n*m, c].reshape(n, m)
        V_c = V[:m*p, c].reshape(m, p)
        W_c = W[:p*n, c].reshape(p, n)
        
        u_a = np.sum(U_c * A, axis=1)  # shape (n,)
        b_v = np.sum(B * V_c, axis=0)  # shape (p,)
        C_comp += W_c.T * np.outer(u_a, b_v)
    
    return np.allclose(C_comp, C_true, atol=1e-10)

def _block_decompose_4x5x5():
    """Decompose 4x5x5 using block structure.
    
    Split A (4x5) into A_top (2x5) and A_bot (2x5)
    Split B (5x5) into B_left (5x2) and B_right (5x3)
    
    Sub-problems:
    1. 2x5 * 5x2 -> 2x2 (Strassen, rank 7)
    2. 2x5 * 5x3 -> 2x3 (schoolbook, rank 30)
    3. 2x5 * 5x2 -> 2x2 (Strassen, rank 7)
    4. 2x5 * 5x3 -> 2x3 (schoolbook, rank 30)
    
    Total rank: 74
    """
    n, m, p = 4, 5, 5
    n1, n2 = 2, 2
    p1, p2 = 2, 3
    
    U1, V1, W1 = _strassen_2x2()
    U2, V2, W2 = _schoolbook(2, 5, 3)
    U3, V3, W3 = _strassen_2x2()
    U4, V4, W4 = _schoolbook(2, 5, 3)
    
    total_rank = 7 + 30 + 7 + 30
    U_full = np.zeros((n * m, total_rank), dtype=float)
    V_full = np.zeros((m * p, total_rank), dtype=float)
    W_full = np.zeros((p * n, total_rank), dtype=float)
    
    col_offset = 0
    
    # Sub-problem 1: A_top (rows 0-1) * B_left (cols 0-1)
    for c in range(7):
        for i in range(2):
            for j in range(5):
                U_full[i * 5 + j, col_offset + c] = U1[i * 5 + j, c]
        for j in range(5):
            for k in range(2):
                V_full[j * 5 + k, col_offset + c] = V1[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W_full[k * 4 + i, col_offset + c] = W1[k * 2 + i, c]
    col_offset += 7
    
    # Sub-problem 2: A_top (rows 0-1) * B_right (cols 2-4)
    for c in range(30):
        for i in range(2):
            for j in range(5):
                U_full[i * 5 + j, col_offset + c] = U2[i * 5 + j, c]
        for j in range(5):
            for k in range(3):
                V_full[j * 5 + (2 + k), col_offset + c] = V2[j * 5 + k, c]
        for k in range(3):
            for i in range(2):
                W_full[(2 + k) * 4 + i, col_offset + c] = W2[k * 2 + i, c]
    col_offset += 30
    
    # Sub-problem 3: A_bot (rows 2-3) * B_left (cols 0-1)
    for c in range(7):
        for i in range(2):
            for j in range(5):
                U_full[(2 + i) * 5 + j, col_offset + c] = U3[i * 5 + j, c]
        for j in range(5):
            for k in range(2):
                V_full[j * 5 + k, col_offset + c] = V3[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W_full[k * 4 + (2 + i), col_offset + c] = W3[k * 2 + i, c]
    col_offset += 7
    
    # Sub-problem 4: A_bot (rows 2-3) * B_right (cols 2-4)
    for c in range(30):
        for i in range(2):
            for j in range(5):
                U_full[(2 + i) * 5 + j, col_offset + c] = U4[i * 5 + j, c]
        for j in range(5):
            for k in range(3):
                V_full[j * 5 + (2 + k), col_offset + c] = V4[j * 5 + k, c]
        for k in range(3):
            for i in range(2):
                W_full[(2 + k) * 4 + (2 + i), col_offset + c] = W4[k * 2 + i, c]
    col_offset += 30
    
    assert col_offset == total_rank
    return U_full, V_full, W_full

def _try_3x5x5_block():
    """Try to decompose 3x5x5 using 2x5x5 + 1x5x5 blocks.
    
    2x5x5: can use Strassen-like or schoolbook (rank 50)
    1x5x5: schoolbook rank 25
    Total: 75 (worse than 74)
    
    Or split differently:
    3x5x5: split rows as 1+2
    1x5 * 5x5 -> 1x5: schoolbook rank 25
    2x5 * 5x5 -> 2x5: schoolbook rank 50
    Total: 75
    """
    pass

def _try_4x5x5_various_splits():
    """Try various block splits for 4x5x5 and return the best one."""
    n, m, p = 4, 5, 5
    best_rank = n * m * p
    best_result = None
    
    # Split A rows and B cols
    for n1 in range(1, n):
        n2 = n - n1
        for p1 in range(1, p):
            p2 = p - p1
            
            # Sub-problem ranks (schoolbook)
            r1 = n1 * m * p1
            r2 = n1 * m * p2
            r3 = n2 * m * p1
            r4 = n2 * m * p2
            total = r1 + r2 + r3 + r4
            
            # Use Strassen for 2x2 blocks where possible
            if n1 == 2 and p1 == 2:
                total -= (r1 - 7)
            if n2 == 2 and p1 == 2:
                total -= (r3 - 7)
            if n1 == 2 and p2 == 2:
                total -= (r2 - 7)
            if n2 == 2 and p2 == 2:
                total -= (r4 - 7)
            
            if total < best_rank:
                best_rank = total
                best_result = (n1, p1)
    
    return best_rank, best_result

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build block decomposition (rank 74)
    U_blk, V_blk, W_blk = _block_decompose_4x5x5()
    
    if _verify_decomposition(U_blk, V_blk, W_blk, n, m, p):
        return {"U": U_blk.tolist(), "V": V_blk.tolist(), "W": W_blk.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
