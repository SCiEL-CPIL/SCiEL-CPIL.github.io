import numpy as np

DIMENSIONS = [4, 5, 6]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7) -> indices i*2+j
    V shape: (4, 7) -> indices j*2+k
    W shape: (4, 7) -> indices k*2+i
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Indices in 2x2 matrix: 00, 01, 10, 11 -> 0, 1, 2, 3
    # U: A[i,j]
    # V: B[j,k]
    # W: C[k,i]
    
    # M1 = (A00 + A11) * (B10 + B11)
    # A00=0, A11=2. B10=2, B11=3.
    # C00 += M1, C11 += M1
    
    # M2 = (A00 + A01) * B11
    # A00=0, A01=1. B11=3.
    # C00 += M2, C11 -= M2
    
    # M3 = A00 * (B01 - B11)
    # A00=0. B01=1, B11=3.
    # C00 -= M3, C11 += M3
    
    # M4 = A11 * (B00 - B01)
    # A11=2. B00=0, B01=1.
    # C00 += M4, C11 += M4
    
    # M5 = (A00 + A10) * B00
    # A00=0, A10=2. B00=0.
    # C10 += M5
    
    # M6 = (A10 + A11) * (B00 + B01)
    # A10=2, A11=3. B00=0, B01=1.
    # C10 += M6, C11 += M6
    
    # M7 = (A00 - A10) * (B10 + B11)
    # A00=0, A10=2. B10=2, B11=3.
    # C01 += M7
    
    # Mapping to columns 0..6
    
    # Col 0 (M1)
    U[0][0] = 1; U[2][0] = 1
    V[2][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[2][0] = 1 # C00, C11
    
    # Col 1 (M2)
    U[0][1] = 1; U[1][1] = 1
    V[3][1] = 1
    W[0][1] = 1; W[2][1] = -1 # C00, C11
    
    # Col 2 (M3)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = -1; W[2][2] = 1 # C00, C11
    
    # Col 3 (M4)
    U[2][3] = 1
    V[0][3] = 1; V[1][3] = -1
    W[0][3] = 1; W[2][3] = 1 # C00, C11
    
    # Col 4 (M5)
    U[0][4] = 1; U[2][4] = 1
    V[0][4] = 1
    W[1][4] = 1 # C10
    
    # Col 5 (M6)
    U[2][5] = 1; U[3][5] = 1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1; W[3][5] = 1 # C10, C11
    
    # Col 6 (M7)
    U[0][6] = 1; U[2][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1 # C01
    
    return U, V, W

def _kronecker_decomp(U1, V1, W1):
    """
    Computes the Kronecker product decomposition for 4x4x4.
    Input: 2x2x2 decomp.
    Output: 4x4x4 decomp.
    """
    # Rank is 7*7 = 49
    # Dimensions are 4x4x4
    # U2 shape (16, 49), V2 (16, 49), W2 (16, 49)
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U[i1*m2 + j1*m2 + j2] ... 
            # Index in 4x4: i = 2*i1 + i2, j = 2*j1 + j2
            # Index in U: i*m + j = (2*i1+i2)*4 + (2*j1+j2)
            
            # U1 contributes to i1, j1. U2 (same as U1) to i2, j2.
            # Actually, Kronecker product of tensors:
            # T[i,j,k] = sum U[i,a] V[j,a] W[k,a]
            # T_kron[i1,i2, j1,j2, k1,k2] = sum U1[i1,a] U2[i2,a'] V1[j1,a] V2[j2,a'] W1[k1,a] W2[k2,a']
            # This can be seen as a single tensor with index (a, a').
            
            for i1 in range(2):
                for j1 in range(2):
                    val1 = U1[i1*2+j1][c1]
                    if val1 != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                val2 = U1[i2*2+j2][c2] # Using same structure for 2nd part
                                if val2 != 0:
                                    idx_u = (i1*2 + i2)*4 + (j1*2 + j2)
                                    U[idx_u][c] += val1 * val2
                                    
            for j1 in range(2):
                for k1 in range(2):
                    val1 = V1[j1*2+k1][c1]
                    if val1 != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                val2 = V1[j2*2+k2][c2]
                                if val2 != 0:
                                    idx_v = (j1*2 + j2)*4 + (k1*2 + k2)
                                    V[idx_v][c] += val1 * val2
                                    
            for k1 in range(2):
                for i1 in range(2):
                    val1 = W1[k1*2+i1][c1]
                    if val1 != 0:
                        for k2 in range(2):
                            for i2 in range(2):
                                val2 = W1[k2*2+i2][c2]
                                if val2 != 0:
                                    idx_w = (k1*2 + k2)*4 + (i1*2 + i2)
                                    W[idx_w][c] += val1 * val2
                                    
    return U, V, W

def _schoolbook_decomp(n, m, p):
    """
    Schoolbook decomposition for n x m x p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _pad_decomp(U, V, W, n, m, p, i_offset, j_offset, k_offset, i_len, j_len, k_len):
    """
    Pads a decomposition of size i_len x j_len x k_len to n x m x p.
    i_offset, j_offset, k_offset are the starting indices in the full matrices.
    """
    full_n, full_m, full_p = n, m, p
    rank = len(U[0])
    
    U_full = [[0]*rank for _ in range(full_n * full_m)]
    V_full = [[0]*rank for _ in range(full_m * full_p)]
    W_full = [[0]*rank for _ in range(full_p * full_n)]
    
    for i in range(i_len):
        for j in range(j_len):
            u_idx = i * j_len + j # Index in local U
            u_full_idx = (i_offset + i) * full_m + (j_offset + j)
            U_full[u_full_idx] = U[u_idx]
            
    for j in range(j_len):
        for k in range(k_len):
            v_idx = j * k_len + k
            v_full_idx = (j_offset + j) * full_p + (k_offset + k)
            V_full[v_full_idx] = V[v_idx]
            
    for k in range(k_len):
        for i in range(i_len):
            w_idx = k * i_len + i
            w_full_idx = (k_offset + k) * full_n + (i_offset + i)
            W_full[w_full_idx] = W[w_idx]
            
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Core 4x4x4 (Strassen)
    U_core_local, V_core_local, W_core_local = _kronecker_decomp(*_strassen_2x2x2())
    U_core, V_core, W_core = _pad_decomp(U_core_local, V_core_local, W_core_local, n, m, p, 0, 0, 0, 4, 4, 4)
    
    # 2. Rem1: 4x4x2 (A_core * B_rem_cols)
    # A: 4x4 (0,0), B: 4x2 (0,4), C: 4x2 (0,4)
    U_rem1_local, V_rem1_local, W_rem1_local = _schoolbook_decomp(4, 4, 2)
    U_rem1, V_rem1, W_rem1 = _pad_decomp(U_rem1_local, V_rem1_local, W_rem1_local, n, m, p, 0, 0, 4, 4, 4, 2)
    
    # 3. Rem2: 4x1x4 (A_rem * B_rem_rows)
    # A: 4x1 (0,4), B: 1x4 (4,0), C: 4x4 (0,0)
    U_rem2_local, V_rem2_local, W_rem2_local = _schoolbook_decomp(4, 1, 4)
    U_rem2, V_rem2, W_rem2 = _pad_decomp(U_rem2_local, V_rem2_local, W_rem2_local, n, m, p, 0, 4, 0, 4, 1, 4)
    
    # 4. Rem3: 4x1x2 (A_rem * B_corner)
    # A: 4x1 (0,4), B: 1x2 (4,4), C: 4x2 (0,4)
    U_rem3_local, V_rem3_local, W_rem3_local = _schoolbook_decomp(4, 1, 2)
    U_rem3, V_rem3, W_rem3 = _pad_decomp(U_rem3_local, V_rem3_local, W_rem3_local, n, m, p, 0, 4, 4, 4, 1, 2)
    
    # Concatenate
    U = [row[:] for row in U_core]
    V = [row[:] for row in V_core]
    W = [row[:] for row in W_core]
    
    for i in range(len(U)):
        for j in range(len(U[0])):
            U[i][j] += U_rem1[i][j] + U_rem2[i][j] + U_rem3[i][j]
            V[i][j] += V_rem1[i][j] + V_rem2[i][j] + V_rem3[i][j]
            W[i][j] += W_rem1[i][j] + W_rem2[i][j] + W_rem3[i][j]
            
    return {"U": U, "V": V, "W": W}
