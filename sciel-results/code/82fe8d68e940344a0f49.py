"""
Rank-32 tensor decomposition for 2x2 * 2x9 matrix multiplication.

Uses Strassen's rank-7 algorithm on four 2x2 * 2x2 column pairs,
and schoolbook rank-4 on the remaining single column.
Total rank = 4*7 + 4 = 32.
"""

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-32 exact tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    Index conventions (trace convention for W):
    - U: shape (n*m, rank) = (4, rank), indexed by (i*m + j), i in {0,1}, j in {0,1}
    - V: shape (m*p, rank) = (18, rank), indexed by (j*p + k), j in {0,1}, k in {0..8}
    - W: shape (p*n, rank) = (18, rank), indexed by (k*n + i), k in {0..8}, i in {0,1}
    
    The tensor T satisfies:
      T[(i,j), (j,k), (k,i)] = 1 for all valid i,j,k
      and 0 otherwise.
    
    Decomposition: sum over r of U[:,r] * V[:,r] * W[:,r] = T
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 32
    V = [[0] * rank for _ in range(m * p)]   # 18 x 32
    W = [[0] * rank for _ in range(p * n)]   # 18 x 32
    
    def strassen_block(k1, k2, start_col):
        """
        Add 7 Strassen products for a 2x2 * 2x2 block with columns k1, k2.
        A is 2x2 (rows i=0,1; cols j=0,1), B is 2x2 (rows j=0,1; cols k=k1,k2).
        """
        # V row indices: j*p + k
        v_00 = 0 * p + k1  # B[0, k1]
        v_01 = 0 * p + k2  # B[0, k2]
        v_10 = 1 * p + k1  # B[1, k1]
        v_11 = 1 * p + k2  # B[1, k2]
        
        # W row indices: k*n + i
        w_00 = k1 * n + 0  # C[0, k1]
        w_01 = k1 * n + 1  # C[1, k1]
        w_02 = k2 * n + 0  # C[0, k2]
        w_03 = k2 * n + 1  # C[1, k2]
        
        # U row indices: i*m + j
        u_00 = 0  # A[0,0]
        u_01 = 1  # A[0,1]
        u_10 = 2  # A[1,0]
        u_11 = 3  # A[1,1]
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,k1] + B[1,k2])
        c = start_col + 0
        U[u_00][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        W[w_03][c] += 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,k1]
        c = start_col + 1
        U[u_10][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        W[w_01][c] += 1
        W[w_03][c] -= 1
        
        # M3 = A[0,0] * (B[0,k2] - B[1,k2])
        c = start_col + 2
        U[u_00][c] += 1
        V[v_01][c] += 1
        V[v_11][c] -= 1
        W[w_02][c] += 1
        W[w_03][c] += 1
        
        # M4 = A[1,1] * (B[1,k1] - B[0,k1])
        c = start_col + 3
        U[u_11][c] += 1
        V[v_10][c] += 1
        V[v_00][c] -= 1
        W[w_00][c] += 1
        W[w_01][c] += 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,k2]
        c = start_col + 4
        U[u_00][c] += 1
        U[u_01][c] += 1
        V[v_11][c] += 1
        W[w_00][c] -= 1
        W[w_02][c] += 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,k1] + B[0,k2])
        c = start_col + 5
        U[u_10][c] += 1
        U[u_00][c] -= 1
        V[v_00][c] += 1
        V[v_01][c] += 1
        W[w_03][c] += 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,k1] + B[1,k2])
        c = start_col + 6
        U[u_01][c] += 1
        U[u_11][c] -= 1
        V[v_10][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        
        return start_col + 7
    
    def schoolbook_single(k, start_col):
        """
        Add 4 schoolbook products for a single column k.
        """
        v_0 = 0 * p + k  # B[0, k]
        v_1 = 1 * p + k  # B[1, k]
        
        w_0 = k * n + 0  # C[0, k]
        w_1 = k * n + 1  # C[1, k]
        
        u_00 = 0  # A[0,0]
        u_01 = 1  # A[0,1]
        u_10 = 2  # A[1,0]
        u_11 = 3  # A[1,1]
        
        # N1 = A[0,0] * B[0,k]
        c = start_col + 0
        U[u_00][c] += 1
        V[v_0][c] += 1
        W[w_0][c] += 1
        
        # N2 = A[0,1] * B[1,k]
        c = start_col + 1
        U[u_01][c] += 1
        V[v_1][c] += 1
        W[w_0][c] += 1
        
        # N3 = A[1,0] * B[0,k]
        c = start_col + 2
        U[u_10][c] += 1
        V[v_0][c] += 1
        W[w_1][c] += 1
        
        # N4 = A[1,1] * B[1,k]
        c = start_col + 3
        U[u_11][c] += 1
        V[v_1][c] += 1
        W[w_1][c] += 1
        
        return start_col + 4
    
    col = 0
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = strassen_block(k1, k2, col)
    
    col = schoolbook_single(8, col)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
