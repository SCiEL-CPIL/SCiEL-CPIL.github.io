"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]
TIME_LIMIT = 150.0

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def _build_strassen_2x2():
    """
    Build rank-7 Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22) * (B11 + B22)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # M2 = (A21 + A22) * B11
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # M3 = A11 * (B12 - B22)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[0, 2] = 1.0; W[3, 2] = 1.0
    
    # M4 = A22 * (B21 - B11)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0
    
    # M5 = (A11 + A12) * B22
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[1, 4] = 1.0
    
    # M6 = (A21 - A11) * (B11 + B12)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0
    
    # M7 = (A12 - A22) * (B21 + B22)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_hybrid(n, m, p):
    """
    Build a hybrid decomposition using Strassen for 2x2 blocks where possible,
    and schoolbook for the rest.
    
    For 4x5 * 5x6:
    - Split A into 4x2 and 4x3
    - Split B into 2x6 and 3x6
    - The 4x2 * 2x6 part can use 6 Strassen 2x2 products (rank 7 each = 42)
    - The 4x3 * 3x6 part uses schoolbook (rank 72)
    - Total rank: 42 + 72 = 114
    """
    # Total rank
    total_rank = 114
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Get Strassen 2x2 decomposition
    U_s, V_s, W_s = _build_strassen_2x2()
    
    col = 0
    # Part 1: 4x2 * 2x6 using Strassen
    # A is 4x2 (rows 0-3, cols 0-1), B is 2x6 (rows 0-1, cols 0-5)
    # C is 4x6 (rows 0-3, cols 0-5)
    # We can decompose this as 3 Strassen products along the columns
    # Each 2x2 * 2x2 product has rank 7
    # We need 3 such products to cover 6 columns of B
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    for r in range(7):
        # U: A indices
        U[0 * m + 0, col + r] += U_s[0, r]  # A11
        U[0 * m + 1, col + r] += U_s[1, r]  # A12
        U[1 * m + 0, col + r] += U_s[2, r]  # A21
        U[1 * m + 1, col + r] += U_s[3, r]  # A22
        
        # V: B indices
        V[0 * p + 0, col + r] += V_s[0, r]  # B11
        V[0 * p + 1, col + r] += V_s[1, r]  # B12
        V[1 * p + 0, col + r] += V_s[2, r]  # B21
        V[1 * p + 1, col + r] += V_s[3, r]  # B22
        
        # W: C indices
        W[0 * n + 0, col + r] += W_s[0, r]  # C11
        W[0 * n + 1, col + r] += W_s[1, r]  # C12
        W[1 * n + 0, col + r] += W_s[2, r]  # C21
        W[1 * n + 1, col + r] += W_s[3, r]  # C22
    col += 7
    
    # Block 2: A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    for r in range(7):
        U[2 * m + 0, col + r] += U_s[0, r]
        U[2 * m + 1, col + r] += U_s[1, r]
        U[3 * m + 0, col + r] += U_s[2, r]
        U[3 * m + 1, col + r] += U_s[3, r]
        
        V[0 * p + 0, col + r] += V_s[0, r]
        V[0 * p + 1, col + r] += V_s[1, r]
        V[1 * p + 0, col + r] += V_s[2, r]
        V[1 * p + 1, col + r] += V_s[3, r]
        
        W[0 * n + 2, col + r] += W_s[0, r]
        W[0 * n + 3, col + r] += W_s[1, r]
        W[1 * n + 2, col + r] += W_s[2, r]
        W[1 * n + 3, col + r] += W_s[3, r]
    col += 7
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    for r in range(7):
        U[0 * m + 0, col + r] += U_s[0, r]
        U[0 * m + 1, col + r] += U_s[1, r]
        U[1 * m + 0, col + r] += U_s[2, r]
        U[1 * m + 1, col + r] += U_s[3, r]
        
        V[0 * p + 2, col + r] += V_s[0, r]
        V[0 * p + 3, col + r] += V_s[1, r]
        V[1 * p + 2, col + r] += V_s[2, r]
        V[1 * p + 3, col + r] += V_s[3, r]
        
        W[2 * n + 0, col + r] += W_s[0, r]
        W[2 * n + 1, col + r] += W_s[1, r]
        W[3 * n + 0, col + r] += W_s[2, r]
        W[3 * n + 1, col + r] += W_s[3, r]
    col += 7
    
    # Block 4: A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4]
    for r in range(7):
        U[2 * m + 0, col + r] += U_s[0, r]
        U[2 * m + 1, col + r] += U_s[1, r]
        U[3 * m + 0, col + r] += U_s[2, r]
        U[3 * m + 1, col + r] += U_s[3, r]
        
        V[0 * p + 2, col + r] += V_s[0, r]
        V[0 * p + 3, col + r] += V_s[1, r]
        V[1 * p + 2, col + r] += V_s[2, r]
        V[1 * p + 3, col + r] += V_s[3, r]
        
        W[2 * n + 2, col + r] += W_s[0, r]
        W[2 * n + 3, col + r] += W_s[1, r]
        W[3 * n + 2, col + r] += W_s[2, r]
        W[3 * n + 3, col + r] += W_s[3, r]
    col += 7
    
    # Block 5: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    for r in range(7):
        U[0 * m + 0, col + r] += U_s[0, r]
        U[0 * m + 1, col + r] += U_s[1, r]
        U[1 * m + 0, col + r] += U_s[2, r]
        U[1 * m + 1, col + r] += U_s[3, r]
        
        V[0 * p + 4, col + r] += V_s[0, r]
        V[0 * p + 5, col + r] += V_s[1, r]
        V[1 * p + 4, col + r] += V_s[2, r]
        V[1 * p + 5, col + r] += V_s[3, r]
        
        W[4 * n + 0, col + r] += W_s[0, r]
        W[4 * n + 1, col + r] += W_s[1, r]
        W[5 * n + 0, col + r] += W_s[2, r]
        W[5 * n + 1, col + r] += W_s[3, r]
    col += 7
    
    # Block 6: A[2:4, 0:2] * B[0:2, 4:6] -> C[2:4, 4:6]
    for r in range(7):
        U[2 * m + 0, col + r] += U_s[0, r]
        U[2 * m + 1, col + r] += U_s[1, r]
        U[3 * m + 0, col + r] += U_s[2, r]
        U[3 * m + 1, col + r] += U_s[3, r]
        
        V[0 * p + 4, col + r] += V_s[0, r]
        V[0 * p + 5, col + r] += V_s[1, r]
        V[1 * p + 4, col + r] += V_s[2, r]
        V[1 * p + 5, col + r] += V_s[3, r]
        
        W[4 * n + 2, col + r] += W_s[0, r]
        W[4 * n + 3, col + r] += W_s[1, r]
        W[5 * n + 2, col + r] += W_s[2, r]
        W[5 * n + 3, col + r] += W_s[3, r]
    col += 7
    
    # Part 2: 4x3 * 3x6 using schoolbook
    # A is 4x3 (rows 0-3, cols 2-4), B is 3x6 (rows 2-4, cols 0-5)
    # C is 4x6 (rows 0-3, cols 0-5)
    for i in range(n):
        for j in range(2, m):  # j = 2, 3, 4
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def construct():
    """
    Find a low-rank exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Returns hybrid decomposition (rank 114) using Strassen for 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try hybrid decomposition first
    U, V, W = _build_hybrid(n, m, p)
    
    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
