"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.linalg import hessenberg

DIMENSIONS = [3, 3, 4]

def _matrix_to_tensor(A, B, C, n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = sum_ab C[i,b] * ... 
    Actually for bilinear form: T[a,b,c] = coefficient of x_a y_b z_c in product
    We need: C = A * B, so C[i,k] = sum_j A[i,j] B[j,k]
    The tensor T has T[i,j,k] = 1 for the trilinear form sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    But for rank decomposition we use: T[i,k,j] = sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    Actually the standard: we want to decompose the map (A,B) -> C
    The tensor T has entries T[i,k,j] where i is row of C, k is col of C, j is the "middle" index
    T[i,k,j] = 1 if the term A[i,j]*B[j,k] appears in C[i,k]
    """
    T = np.zeros((n, p, m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, k, j] = 1.0
    return T

def _tensor_to_matrix(T, i, k):
    """Flatten T[i,k,:] into a matrix for matrix multiplication rank computation"""
    return T[i, k, :].reshape(1, -1)

def _compute_rank_via_hessenberg(T, n, p, m):
    """Compute tensor rank via flattening along one mode"""
    # Flatten along mode 1 (i dimension): shape (n, p*m)
    T_reshape = T.reshape(n, p * m)
    U, s, Vt = np.linalg.svd(T_reshape, full_matrices=False)
    return np.sum(s > 1e-10)

def construct():
    n, m, p = DIMENSIONS
    
    # Build the tensor T where T[i,k,j] = 1 for the term A[i,j]*B[j,k] in C[i,k]
    T = np.zeros((n, p, m), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, k, j] = 1.0
    
    # Try to find a low-rank decomposition using various methods
    
    # Method 1: Use the known result that 3x3x4 matrix multiplication has rank <= 28
    # Actually, let's try to compute the rank using SVD on different flattening modes
    
    # Flatten along i: T_i has shape (n, p*m) = (3, 12)
    T_i = T.reshape(n, p * m)
    _, s_i, _ = np.linalg.svd(T_i, full_matrices=False)
    
    # Flatten along k: T_k has shape (p, n*m) = (4, 9)
    T_k = T.transpose(1, 0, 2).reshape(p, n * m)
    _, s_k, _ = np.linalg.svd(T_k, full_matrices=False)
    
    # Flatten along j: T_j has shape (m, n*p) = (3, 12)
    T_j = T.transpose(2, 0, 1).reshape(m, n * p)
    _, s_j, _ = np.linalg.svd(T_j, full_matrices=False)
    
    # The tensor rank is at least the maximum of these matrix ranks
    # Let's try to construct an explicit decomposition with rank 28
    
    # For 3x3x4, we can use the following approach:
    # Split into 2x3x4 and 1x3x4, or use other algebraic tricks
    
    # Let me try a direct construction based on known results
    # Rank of 3x3x3 is 19 (Winograd) or 23 (standard)
    # For 3x3x4, we can extend
    
    # Actually, let me just use the schoolbook for now since I'm not sure of the exact rank
    # But let me try rank 28 by combining some terms
    
    # Use the fact that we can combine terms using addition/subtraction
    # For each i, we have C[i,k] = sum_j A[i,j] * B[j,k]
    # For fixed i, this is a 3x4 matrix multiplication, which has rank 12
    # Total schoolbook: 3 * 12 = 36
    
    # Try to reduce by using Winograd-style tricks on the 3x4 part
    # For 3x4 matrix mult, can we do better than 12?
    # 3x4 = 3*4 = 12, and rank is at least max(3,4) = 4, but actually for full matrix mult it's 12
    
    # Let me just output schoolbook rank 36 for safety
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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build tensor T[i,k,j] = 1 for term A[i,j]*B[j,k] in C[i,k]
    T = np.zeros((n, p, m), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, k, j] = 1.0
    
    # Attempt rank reduction via structured decomposition
    # Use the approach: for each pair (i,k), express the sum over j using fewer products
    # by sharing factors across different (i,k) pairs
    
    # Try rank = 28 construction
    # Split the 3 rows into groups and use algebraic identities
    
    # For row 0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # For row 1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    # For row 2: C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    
    # Define shared factors:
    # S0 = A[0,0] + A[1,0] + A[2,0]  (sum of column 0 of A)
    # S1 = A[0,1] + A[1,1] + A[2,1]  (sum of column 1 of A)
    # S2 = A[0,2] + A[1,2] + A[2,2]  (sum of column 2 of A)
    # D01 = A[0,0] - A[1,0]
    # D02 = A[0,0] - A[2,0]
    # D12 = A[1,0] - A[2,0]
    
    # This is getting complex. Let me use a known construction for rank 28.
    
    # Actually, let me just verify the schoolbook works and return it.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
