import numpy as np

def _build_part1():
    """
    Build decomposition for N=1, M=2, P=3.
    Rank = 6.
    U: (1*2, 6) = (2, 6)
    V: (2*3, 6) = (6, 6)
    W: (3*1, 6) = (3, 6)
    """
    N, M, P = 1, 2, 3
    R = N * M * P
    U = np.zeros((N * M, R), dtype=float)
    V = np.zeros((M * P, R), dtype=float)
    W = np.zeros((P * N, R), dtype=float)
    
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, c] = 1
                V[j * P + k, c] = 1
                W[k * N + i, c] = 1
                c += 1
    return U, V, W

def _build_part2():
    """
    Build decomposition for N=3, M=2, P=2.
    Rank = 11.
    Split N=3 into 1+2.
    Part 2a: N=1, M=2, P=2. Rank 4.
    Part 2b: N=2, M=2, P=2. Rank 7.
    
    U: (3*2, 11) = (6, 11)
    V: (2*2, 11) = (4, 11)
    W: (2*3, 11) = (6, 11)
    """
    N, M, P = 3, 2, 2
    R = 11
    U = np.zeros((N * M, R), dtype=float)
    V = np.zeros((M * P, R), dtype=float)
    W = np.zeros((P * N, R), dtype=float)
    
    # Part 2a: N_a=1, M_a=2, P_a=2. Rank 4.
    # A_a is 1x2, B_a is 2x2. C_a is 1x2.
    # U_a: (1*2, 4) = (2, 4)
    # V_a: (2*2, 4) = (4, 4)
    # W_a: (2*1, 4) = (2, 4)
    
    # Part 2b: N_b=2, M_b=2, P_b=2. Rank 7.
    # A_b is 2x2, B_b is 2x2. C_b is 2x2.
    # U_b: (2*2, 7) = (4, 7)
    # V_b: (2*2, 7) = (4, 7)
    # W_b: (2*2, 7) = (4, 7)
    
    # We need to map these into the global U, V, W for N=3, M=2, P=2.
    # Global A is 3x2.
    # A = [A_a; A_b].
    # Rows 0 of A correspond to A_a.
    # Rows 1,2 of A correspond to A_b.
    
    # Global V is 2x2.
    # V is same for both?
    # C = [A_a B; A_b B].
    # So B is the same B for both.
    # B is 2x2.
    # V_b is the V for 2x2x2.
    # V_a is the V for 1x2x2.
    # Since B is the same, we can use the same V columns?
    # No, the products are different.
    # In Part 2a, we compute A_a B.
    # In Part 2b, we compute A_b B.
    # The V matrix for Part 2a has entries V_a[j, c].
    # The V matrix for Part 2b has entries V_b[j, c].
    # Since B is the same, and the decomposition is linear,
    # we can just take the V from Part 2b?
    # No, Part 2a uses 4 products, Part 2b uses 7 products.
    # The V for Part 2a is a 4x4 identity-like matrix (schoolbook).
    # The V for Part 2b is the Strassen V.
    # We need to combine them.
    
    # Let's build U_2a, V_2a, W_2a for 1x2x2.
    U_2a, V_2a, W_2a = _build_part1_sub(1, 2, 2)
    # Let's build U_2b, V_2b, W_2b for 2x2x2.
    U_2b, V_2b, W_2b = _build_strassen_2x2x2()
    
    # Map to Global U (6x11)
    # U_2a is 2x4. Rows 0,1 of Global U.
    U[0:2, 0:4] = U_2a
    # U_2b is 4x7. Rows 2,3,4,5 of Global U.
    U[2:6, 4:11] = U_2b
    
    # Map to Global V (4x11)
    # V_2a is 4x4.
    V[:, 0:4] = V_2a
    # V_2b is 4x7.
    V[:, 4:11] = V_2b
    
    # Map to Global W (6x11)
    # W_2a is 2x4. (P*N = 2*1 = 2).
    # W_2b is 4x7. (P*N = 2*2 = 4).
    # Global W is 6x11.
    # Rows of Global W correspond to (k, i) where i in 0..2, k in 0..1.
    # W index = k * 3 + i.
    # For Part 2a (i=0): k in 0..1.
    # W_2a[k, c] maps to W[k * 3 + 0, c].
    for k in range(2):
        W[k * 3 + 0, 0:4] = W_2a[k, :]
        
    # For Part 2b (i=1,2): k in 0..1.
    # W_2b[k, i_b, c] where i_b in 0..1.
    # Global i = 1 + i_b.
    # W index = k * 3 + (1 + i_b).
    for k in range(2):
        for i_b in range(2):
            W[k * 3 + (1 + i_b), 4:11] = W_2b[k * 2 + i_b, :]
            
    return U, V, W

def _build_part1_sub(N, M, P):
    """Schoolbook for small N, M, P."""
    R = N * M * P
    U = np.zeros((N * M, R), dtype=float)
    V = np.zeros((M * P, R), dtype=float)
    W = np.zeros((P * N, R), dtype=float)
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, c] = 1
                V[j * P + k, c] = 1
                W[k * N + i, c] = 1
                c += 1
    return U, V, W

def _build_strassen_2x2x2():
    """Strassen 2x2x2 rank-7 decomposition."""
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # P1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # P2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # P3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = -1
    
    # P4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[3][3] = 1; W[2][3] = -1
    
    # P5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # P6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1; W[0][5] = -1; W[1][5] = -1
    
    # P7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[1][6] = 1; W[2][6] = 1; W[3][6] = 1
    
    return U, V, W

def construct():
    # Part 1: 1x2x3
    U1, V1, W1 = _build_part1()
    R1 = U1.shape[1]
    N1, M1, P1 = 1, 2, 3
    
    # Part 2: 3x2x2
    U2, V2, W2 = _build_part2()
    R2 = U2.shape[1]
    N2, M2, P2 = 3, 2, 2
    
    N = N1 * N2
    M = M1 * M2
    P = P1 * P2
    R = R1 * R2
    
    U = np.zeros((N * M, R), dtype=float)
    V = np.zeros((M * P, R), dtype=float)
    W = np.zeros((P * N, R), dtype=float)
    
    for c1 in range(R1):
        for c2 in range(R2):
            c = c1 * R2 + c2
            # U
            for i1 in range(N1):
                for i2 in range(N2):
                    i = i1 * N2 + i2
                    # U1 index: i1 * M1 + j1? No, U1 is (N1*M1, R1).
                    # We need to sum over j1, j2?
                    # No, U1[i1*M1+j1, c1] * U2[i2*M2+j2, c2]
                    # This requires summing over j1, j2?
                    # The tensor product of U vectors is not just element-wise.
                    # U_global[i, j, c] = U1[i1, j1, c1] U2[i2, j2, c2]
                    # But U is a matrix (NM, R).
                    # The row index of U is i.
                    # Wait, U is not a 3D tensor in the output, it's a matrix.
                    # The "rank-1 term" is U_col * V_col * W_col.
                    # U_col is a vector of length NM.
                    # U_col[i] = U1[i1, j1] * U2[i2, j2]?
                    # No.
                    # A = A1 x A2.
                    # A[i, j] = A1[i1, j1] A2[i2, j2].
                    # The decomposition of A is sum_c U1[i1, j1, c1] U2[i2, j2, c2].
                    # So U_global[i, c] is not simply defined by i alone.
                    # U_global is a matrix.
                    # The rank-1 tensor is u v^T w^T.
                    # u is a vector of length NM.
                    # u[i] = U1[i1, j1] U2[i2, j2] ?
                    # No, j is the column of A.
                    # The vector u corresponds to the flattened A.
                    # u[i*M+j] = A[i,j].
                    # So u[i*M+j] = U1[i1*M1+j1, c1] * U2[i2*M2+j2, c2].
                    
                    # So we iterate over all i1, j1, i2, j2.
                    pass
                    
    # Correct implementation:
    U = np.zeros((N * M, R), dtype=float)
    V = np.zeros((M * P, R), dtype=float)
    W = np.zeros((P * N, R), dtype=float)
    
    for c1 in range(R1):
        for c2 in range(R2):
            c = c1 * R2 + c2
            # U
            for i1 in range(N1):
                for j1 in range(M1):
                    for i2 in range(N2):
                        for j2 in range(M2):
                            i = i1 * N2 + i2
                            j = j1 * M2 + j2
                            val = U1[i1 * M1 + j1, c1] * U2[i2 * M2 + j2, c2]
                            U[i * M + j, c] = val
                            
            # V
            for j1 in range(M1):
                for k1 in range(P1):
                    for j2 in range(M2):
                        for k2 in range(P2):
                            j = j1 * M2 + j2
                            k = k1 * P2 + k2
                            val = V1[j1 * P1 + k1, c1] * V2[j2 * P2 + k2, c2]
                            V[j * P + k, c] = val
                            
            # W
            for k1 in range(P1):
                for i1 in range(N1):
                    for k2 in range(P2):
                        for i2 in range(N2):
                            k = k1 * P2 + k2
                            i = i1 * N2 + i2
                            val = W1[k1 * N1 + i1, c1] * W2[k2 * N2 + i2, c2]
                            W[k * N + i, c] = val
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
