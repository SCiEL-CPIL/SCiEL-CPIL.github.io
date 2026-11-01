"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
import time
import itertools

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i,j,k] = 1 for the product (i,j) of A and B
    # T is indexed by (i, j, k) where C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, c] corresponds to output C[i,k]
    # The decomposition requires: for all i,j,k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if the product term exists
    # Actually the tensor is: T[i, j, k] = 1 (since C[i,k] = sum_j A[i,j]B[j,k])
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, let me re-read the convention.
    # U shape (n*m, rank): indexed by (i, j) -> row i*m+j
    # V shape (m*p, rank): indexed by (j, k) -> row j*p+k
    # W shape (p*n, rank): indexed by (k, i) -> row k*n+i (trace convention W[k*n+i, r])
    # The tensor for matrix multiplication: T[(i,j), (j,k), (k,i)]... 
    # Actually the standard formulation: we want to compute C = A*B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 (it's the identity in the j-sum sense).
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return rank-7 Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    Using standard Strassen formulas.
    """
    # Standard Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U: rows indexed by (i,j) for A[i,j], i,j in {0,1}
    # V: rows indexed by (j,k) for B[j,k], j,k in {0,1}
    # W: rows indexed by (k,i) for C[i,k], i,k in {0,1}
    
    U = [
        # i=0,j=0 (A00): M1, M3, M5, M6
        [1, 0, 1, 0, 1, -1, 0],
        # i=0,j=1 (A01): M5, M7
        [0, 0, 0, 0, 1, 0, 1],
        # i=1,j=0 (A10): M2, M6
        [0, 1, 0, 0, 0, 1, 0],
        # i=1,j=1 (A11): M1, M2, M4, M7
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V = [
        # j=0,k=0 (B00): M1, M2, M4, M6
        [1, 1, 0, -1, 0, 1, 0],
        # j=0,k=1 (B01): M1, M3, M6
        [1, 1, 0, 0, 0, 1, 0],
        # j=1,k=0 (B10): M2, M4, M7
        [0, 1, 0, 1, 0, 0, 1],
        # j=1,k=1 (B11): M1, M3, M5, M7
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W = [
        # k=0,i=0 (C00): M1, M4, M5, M7
        [1, 0, 0, 1, -1, 0, 1],
        # k=1,i=0 (C01): M3, M5
        [0, 0, 1, 0, 1, 0, 0],
        # k=0,i=1 (C10): M2, M4
        [0, 1, 0, 1, 0, 0, 0],
        # k=1,i=1 (C11): M1, M2, M3, M6
        [1, -1, 1, 0, 0, 1, 0],
    ]
    return U, V, W

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x2 times 2x7 matrix multiplication.
    Strategy: Use Strassen (rank 7) for the first 2x2 block of B, and schoolbook (rank 4)
    for the remaining 2x5 part. Total rank = 7 + 4*5 = 27.
    But we can do better: treat it as computing C = A*B where A is 2x2, B is 2x7.
    C is 2x7. C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k].
    
    Alternative: partition B into two 2x2 blocks and one 2x3 block.
    Use Strassen for each 2x2 block (rank 7 each) and schoolbook for 2x3 (rank 6).
    Total: 7+7+6 = 20.
    
    Actually, let's think more carefully. The tensor is T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,...,6}.
    This is a 2 x 2 x 7 tensor.
    
    We can use the fact that this tensor is the "slice" of a larger tensor.
    
    Let me try a direct approach: use bilinear decomposition.
    For each k, C[:,k] = A * B[:,k].
    The full computation is: for each of the 7 output columns, we multiply 2x2 by 2x1.
    
    Schoolbook for 2x2 x 2x1: rank = 2*2*1 = 4 per column, total 7*4 = 28.
    
    Using Strassen for 2x2 x 2x2: we can group columns.
    Group columns (0,1), (2,3), (4,5), and column 6 separately.
    Strassen for each 2x2 group: rank 7 each, so 3*7 = 21.
    Schoolbook for column 6 (2x2 x 2x1): rank 4.
    Total: 21 + 4 = 25.
    
    Can we do better? What if we use a smarter grouping?
    
    Actually, let me just implement the rank-25 decomposition and verify it.
    """
    n, m, p = DIMENSIONS  # 2, 2, 7
    
    # Group columns: (0,1), (2,3), (4,5) use Strassen (rank 7 each), column 6 uses schoolbook (rank 4)
    # Total rank = 7*3 + 4 = 25
    
    total_rank = 7 * 3 + 4
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for block in range(3):
        k_start = block * 2
        sU, sV, sW = _strassen_2x2x2()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m+j][col_offset + r] = sU[i*2+j][r]
            for j in range(2):
                for k in range(2):
                    V[j*p + (k_start + k)][col_offset + r] = sV[j*2+k][r]
            for k in range(2):
                for i in range(2):
                    W[(k_start + k)*n + i][col_offset + r] = sW[k*2+i][r]
        col_offset += 7
    
    # Schoolbook for column 6
    k = 6
    for i in range(n):
        for j in range(m):
            U[i*m+j][col_offset] = 1
            V[j*p+k][col_offset] = 1
            W[k*n+i][col_offset] = 1
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
