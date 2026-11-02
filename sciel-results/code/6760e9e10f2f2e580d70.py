import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-28 tensor decomposition for 2x3 * 3x5 matrix multiplication.
    
    The decomposition uses:
    - Two 2x2x2 Strassen blocks (each rank 7) for the 2x2 * 2x2 parts
    - Schoolbook terms for the remaining parts
    
    Tensor convention: T[i,j,k] where C[i,k] += A[i,j] * B[j,k]
    Factorization: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    """
    n, m, p = 2, 3, 5
    
    # Total rank: 7 + 7 + 4 + 10 = 28
    rank = 28
    
    U = [[0.0] * rank for _ in range(n * m)]   # shape (6, 28)
    V = [[0.0] * rank for _ in range(m * p)]   # shape (15, 28)
    W = [[0.0] * rank for _ in range(p * n)]   # shape (10, 28)
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(i_base, j_base, k_base):
        """
        Add a rank-7 Strassen decomposition for a 2x2x2 matrix multiplication block.
        A indices: rows i_base..i_base+1, cols j_base..j_base+1
        B indices: rows j_base..j_base+1, cols k_base..k_base+1
        C indices: rows i_base..i_base+1, cols k_base..k_base+1
        
        Strassen formulas:
        M1 = (A00 + A11) * (B00 + B11)
        M2 = (A10 + A11) * B00
        M3 = A00 * (B10 - B11)
        M4 = A11 * (B01 - B00)
        M5 = (A00 + A01) * B11
        M6 = (A10 - A00) * (B00 + B10)
        M7 = (A01 - A11) * (B01 + B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        nonlocal col
        
        # Local indices
        i0, i1 = i_base, i_base + 1
        j0, j1 = j_base, j_base + 1
        k0, k1 = k_base, k_base + 1
        
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        U[i0*m + j0][col] = 1.0
        U[i1*m + j1][col] = 1.0
        V[j0*p + k0][col] = 1.0
        V[j1*p + k1][col] = 1.0
        # M1 contributes to: C00 (+1), C11 (+1)
        W[k0*n + i0][col] = 1.0
        W[k1*n + i1][col] = 1.0
        col += 1
        
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        U[i1*m + j0][col] = 1.0
        U[i1*m + j1][col] = 1.0
        V[j0*p + k0][col] = 1.0
        # M2 contributes to: C10 (+1), C11 (-1)
        W[k0*n + i1][col] = 1.0
        W[k1*n + i1][col] = -1.0
        col += 1
        
        # M3 = A[i0,j0] * (B[j1,k0] - B[j1,k1])
        U[i0*m + j0][col] = 1.0
        V[j1*p + k0][col] = 1.0
        V[j1*p + k1][col] = -1.0
        # M3 contributes to: C01 (+1), C11 (+1)
        W[k1*n + i0][col] = 1.0
        W[k1*n + i1][col] = 1.0
        col += 1
        
        # M4 = A[i1,j1] * (B[j0,k1] - B[j0,k0])
        U[i1*m + j1][col] = 1.0
        V[j0*p + k1][col] = 1.0
        V[j0*p + k0][col] = -1.0
        # M4 contributes to: C00 (+1), C10 (+1)
        W[k0*n + i0][col] = 1.0
        W[k0*n + i1][col] = 1.0
        col += 1
        
        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        U[i0*m + j0][col] = 1.0
        U[i0*m + j1][col] = 1.0
        V[j1*p + k1][col] = 1.0
        # M5 contributes to: C00 (-1), C01 (+1)
        W[k0*n + i0][col] = -1.0
        W[k1*n + i0][col] = 1.0
        col += 1
        
        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j1,k0])
        U[i1*m + j0][col] = 1.0
        U[i0*m + j0][col] = -1.0
        V[j0*p + k0][col] = 1.0
        V[j1*p + k0][col] = 1.0
        # M6 contributes to: C11 (+1)
        W[k1*n + i1][col] = 1.0
        col += 1
        
        # M7 = (A[i0,j1] - A[i1,j1]) * (B[j0,k1] + B[j1,k1])
        U[i0*m + j1][col] = 1.0
        U[i1*m + j1][col] = -1.0
        V[j0*p + k1][col] = 1.0
        V[j1*p + k1][col] = 1.0
        # M7 contributes to: C00 (+1)
        W[k0*n + i0][col] = 1.0
        col += 1
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(i_base=0, j_base=0, k_base=0)
    # col should be 7 now
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(i_base=0, j_base=0, k_base=2)
    # col should be 14 now
    
    # Block 3: Schoolbook for A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # 2 * 2 * 1 = 4 terms
    for i in range(2):
        for j in range(2):
            U[i*m + j][col] = 1.0
            V[j*p + 4][col] = 1.0
            W[4*n + i][col] = 1.0
            col += 1
    # col should be 18 now
    
    # Block 4: Schoolbook for A[0:2, 2:3] * B[2:3, 0:5] -> C[0:2, 0:5]
    # 2 * 1 * 5 = 10 terms
    for i in range(2):
        for k in range(5):
            U[i*m + 2][col] = 1.0
            V[2*p + k][col] = 1.0
            W[k*n + i][col] = 1.0
            col += 1
    # col should be 28 now
    
    assert col == 28, f"Expected col=28, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
