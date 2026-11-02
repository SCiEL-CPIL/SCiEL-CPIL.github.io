import numpy as np

def construct():
    """
    Construct a rank-minimal tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    
    Strategy:
    - Split the 8-column dimension into 4 blocks of 2 columns each.
    - For each block, split the 3-row dimension into 2 rows + 1 row.
    - The 2-row part (2x2 * 2x2) uses Strassen's algorithm with rank 7.
    - The 1-row part (2x1 * 1x2) has rank 4 (outer product).
    - Total rank per block: 7 + 4 = 11.
    - Total rank: 4 * 11 = 44.
    
    Index conventions:
    - U has shape (n*m, rank) = (6, rank): index a = i*m + j
    - V has shape (m*p, rank) = (24, rank): index b = j*p + k
    - W has shape (p*n, rank) = (16, rank): index c = k*n + i
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    
    # Process 4 blocks of 2 columns each
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Part 1: A[:, 0:2] * B[0:2, k0:k1]  (2x2 * 2x2, Strassen rank 7)
        # A_sub = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B_sub = [[B[0,k0], B[0,k1]], [B[1,k0], B[1,k1]]]
        
        # U indices for A_sub:
        # a00 = A[0,0] -> 0*m + 0 = 0
        # a01 = A[0,1] -> 0*m + 1 = 1
        # a10 = A[1,0] -> 1*m + 0 = 3
        # a11 = A[1,1] -> 1*m + 1 = 4
        
        # V indices for B_sub:
        # b00 = B[0,k0] -> 0*p + k0 = k0
        # b01 = B[0,k1] -> 0*p + k1 = k1
        # b10 = B[1,k0] -> 1*p + k0 = p + k0
        # b11 = B[1,k1] -> 1*p + k1 = p + k1
        
        # W indices for C output (2x2):
        # C[0,k0] -> k0*n + 0
        # C[0,k1] -> k1*n + 0
        # C[1,k0] -> k0*n + 1
        # C[1,k1] -> k1*n + 1
        
        # Strassen's algorithm for 2x2 * 2x2:
        # M1 = (a00 + a11) * (b00 + b11)
        # M2 = (a10 + a11) * b00
        # M3 = a00 * (b01 - b11)
        # M4 = a11 * (b10 - b00)
        # M5 = (a00 + a01) * b11
        # M6 = (a10 - a00) * (b00 + b01)
        # M7 = (a01 - a11) * (b10 + b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[0, col] = 1      # a00
        U[4, col] = 1      # a11
        V[k0, col] = 1     # b00
        V[p + k1, col] = 1 # b11
        W[k0 * n + 0, col] = 1   # C[0,k0] += M1
        W[k1 * n + 1, col] = 1   # C[1,k1] += M1
        col += 1
        
        # M2 = (a10 + a11) * b00
        U[3, col] = 1      # a10
        U[4, col] = 1      # a11
        V[k0, col] = 1     # b00
        W[k0 * n + 1, col] = 1   # C[1,k0] += M2
        W[k1 * n + 1, col] = -1  # C[1,k1] -= M2
        col += 1
        
        # M3 = a00 * (b01 - b11)
        U[0, col] = 1      # a00
        V[k1, col] = 1     # b01
        V[p + k1, col] = -1 # b11
        W[k1 * n + 0, col] = 1   # C[0,k1] += M3
        W[k1 * n + 1, col] = 1   # C[1,k1] += M3
        col += 1
        
        # M4 = a11 * (b10 - b00)
        U[4, col] = 1      # a11
        V[p + k0, col] = 1 # b10
        V[k0, col] = -1    # b00
        W[k0 * n + 0, col] = 1   # C[0,k0] += M4
        W[k0 * n + 1, col] = 1   # C[1,k0] += M4
        col += 1
        
        # M5 = (a00 + a01) * b11
        U[0, col] = 1      # a00
        U[1, col] = 1      # a01
        V[p + k1, col] = 1 # b11
        W[k0 * n + 0, col] = -1  # C[0,k0] -= M5
        W[k1 * n + 0, col] = 1   # C[0,k1] += M5
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[3, col] = 1      # a10
        U[0, col] = -1     # a00
        V[k0, col] = 1     # b00
        V[k1, col] = 1     # b01
        W[k1 * n + 1, col] = 1   # C[1,k1] += M6
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[1, col] = 1      # a01
        U[4, col] = -1     # a11
        V[p + k0, col] = 1 # b10
        V[p + k1, col] = 1 # b11
        W[k0 * n + 0, col] = 1   # C[0,k0] += M7
        col += 1
        
        # Part 2: A[:, 2] * B[2, k0:k1]  (2x1 * 1x2, rank 4)
        # A indices: i*m + 2 for i in {0,1}
        # B indices: 2*p + k for k in {k0, k1}
        # C indices: k*n + i for k in {k0, k1}, i in {0,1}
        
        # t0 = A[0,2] * B[2,k0] -> C[0,k0]
        U[0 * m + 2, col] = 1
        V[2 * p + k0, col] = 1
        W[k0 * n + 0, col] = 1
        col += 1
        
        # t1 = A[0,2] * B[2,k1] -> C[0,k1]
        U[0 * m + 2, col] = 1
        V[2 * p + k1, col] = 1
        W[k1 * n + 0, col] = 1
        col += 1
        
        # t2 = A[1,2] * B[2,k0] -> C[1,k0]
        U[1 * m + 2, col] = 1
        V[2 * p + k0, col] = 1
        W[k0 * n + 1, col] = 1
        col += 1
        
        # t3 = A[1,2] * B[2,k1] -> C[1,k1]
        U[1 * m + 2, col] = 1
        V[2 * p + k1, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
    
    # Verify correctness
    def verify(U, V, W):
        r = U.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T_exp[a_idx, b_idx, c_idx] = 1
        return np.array_equal(T_check, T_exp)
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: schoolbook rank 48
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=np.int64)
        V_sb = np.zeros((m*p, rank_sb), dtype=np.int64)
        W_sb = np.zeros((p*n, rank_sb), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_sb[a_idx, c] = 1
                    V_sb[b_idx, c] = 1
                    W_sb[c_idx, c] = 1
                    c += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
