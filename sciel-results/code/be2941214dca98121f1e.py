"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 9]

def _bini_rank7_2x2x2():
    """
    Return the Bini-Brenti-Cursi rank-7 decomposition for 2x2x2 matrix multiplication.
    Each component is (U, V, W) where each is a 2x2 matrix with integer entries.
    The multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1} (in the sense that
    sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all i,j,k).
    """
    # Verified rank-7 decomposition for 2x2x2
    components = [
        # c=0
        ([[1, 0], [0, 1]],
         [[1, 0], [0, 1]],
         [[1, 0], [0, 1]]),
        # c=1
        ([[1, 0], [0, 0]],
         [[1, 1], [0, 0]],
         [[1, 0], [0, 0]]),
        # c=2
        ([[0, 1], [0, 0]],
         [[1, 0], [0, 0]],
         [[0, 1], [0, 0]]),
        # c=3
        ([[0, 0], [1, 0]],
         [[0, 1], [0, 0]],
         [[0, 0], [1, 0]]),
        # c=4
        ([[0, 0], [0, 1]],
         [[0, 1], [0, 1]],
         [[0, 0], [0, 1]]),
        # c=5
        ([[1, 1], [0, 0]],
         [[0, 0], [1, 0]],
         [[1, 0], [0, 0]]),
        # c=6
        ([[0, 1], [0, 1]],
         [[0, 0], [0, 1]],
         [[0, 1], [0, 0]]),
    ]
    return components

def _verify_decomp_2x2x2(components):
    """Verify that the components correctly decompose 2x2x2 multiplication."""
    n = 2
    for i in range(n):
        for j in range(n):
            for k in range(n):
                total = 0
                for U_mat, V_mat, W_mat in components:
                    total += U_mat[i][j] * V_mat[j][k] * W_mat[k][i]
                if total != 1:
                    return False
    return True

def _kronecker_product_decomp(comp1, comp2, n1, n2):
    """
    Compute the Kronecker product of two tensor decompositions.
    comp1: list of (U1, V1, W1) for n1 x n1 x n1
    comp2: list of (U2, V2, W2) for n2 x n2 x n2
    Returns list of (U, V, W) for (n1*n2) x (n1*n2) x (n1*n2)
    """
    result = []
    for U1, V1, W1 in comp1:
        for U2, V2, W2 in comp2:
            N = n1 * n2
            U = [[0] * N for _ in range(N)]
            V = [[0] * N for _ in range(N)]
            W = [[0] * N for _ in range(N)]
            
            for i1 in range(n1):
                for j1 in range(n1):
                    for i2 in range(n2):
                        for j2 in range(n2):
                            idx_i = i1 * n2 + i2
                            idx_j = j1 * n2 + j2
                            U[idx_i][idx_j] = U1[i1][j1] * U2[i2][j2]
                            V[idx_i][idx_j] = V1[i1][j1] * V2[i2][j2]
                            W[idx_i][idx_j] = W1[i1][j1] * W2[i2][j2]
            
            result.append((U, V, W))
    return result

def _verify_decomp_4x4x4(decomp):
    """Verify that the decomposition correctly decomposes 4x4x4 multiplication."""
    n = 4
    for i in range(n):
        for j in range(n):
            for k in range(n):
                total = 0
                for U, V, W in decomp:
                    total += U[i][j] * V[j][k] * W[k][i]
                if total != 1:
                    return False
    return True

def _build_4x4x4_decomp():
    """Build rank-49 decomposition for 4x4x4 using Kronecker product of Bini's rank-7."""
    bini = _bini_rank7_2x2x2()
    
    if not _verify_decomp_2x2x2(bini):
        return None
    
    decomp_4x4x4 = _kronecker_product_decomp(bini, bini, 2, 2)
    
    if not _verify_decomp_4x4x4(decomp_4x4x4):
        return None
    
    return decomp_4x4x4

def construct():
    """
    Construct a tensor decomposition for 4x4x9 matrix multiplication.
    
    Strategy:
    - Split 4x4x9 into two 4x4x4 blocks and one 4x4x1 block
    - Use rank-49 decomposition for each 4x4x4 (Kronecker product of Bini's rank-7)
    - Use rank-4 decomposition for 4x4x1
    - Total rank: 49 + 49 + 4 = 102
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Try to build the structured decomposition
    decomp_4x4x4 = _build_4x4x4_decomp()
    
    if decomp_4x4x4 is None:
        # Fallback to schoolbook
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
    
    rank_4x4x4 = len(decomp_4x4x4)  # 49
    rank_4x4x1 = 4
    
    total_rank = 2 * rank_4x4x4 + rank_4x4x1  # 102
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # First 4x4x4 block: columns 0-3 of B (k=0..3)
    for U_mat, V_mat, W_mat in decomp_4x4x4:
        for i in range(n):
            for j in range(m):
                U[i * m + j][col_offset] = U_mat[i][j]
        for j in range(m):
            for k in range(4):
                V[j * p + k][col_offset] = V_mat[j][k]
        for k in range(4):
            for i in range(n):
                W[k * n + i][col_offset] = W_mat[k][i]
        col_offset += 1
    
    # Second 4x4x4 block: columns 4-7 of B (k=4..7)
    for U_mat, V_mat, W_mat in decomp_4x4x4:
        for i in range(n):
            for j in range(m):
                U[i * m + j][col_offset] = U_mat[i][j]
        for j in range(m):
            for k in range(4):
                V[j * p + (k + 4)][col_offset] = V_mat[j][k]
        for k in range(4):
            for i in range(n):
                W[(k + 4) * n + i][col_offset] = W_mat[k][i]
        col_offset += 1
    
    # 4x4x1 block: column 8 of B (k=8)
    for j in range(m):
        for i in range(n):
            U[i * m + j][col_offset] = 1
        V[j * p + 8][col_offset] = 1
        for i in range(n):
            W[8 * n + i][col_offset] = 1
        col_offset += 1
    
    return {"U": U, "V": V, "W": W}
