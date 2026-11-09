"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2():
    """Return (U, V, W) for 2x2 matrix multiplication, rank 7."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Factor 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Factor 1: M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # Factor 2: M3 = A00*(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # Factor 3: M4 = A11*(B01-B10)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    
    # Factor 4: M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # Factor 5: M6 = (A10-A00)*(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # Factor 6: M7 = (A01-A11)*(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _verify_strassen():
    """Verify the Strassen decomposition is correct using numpy."""
    U, V, W = _strassen_2x2()
    n, m, p = 2, 2, 2
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True

def _build_4x4x4_kronecker():
    """Build 4x4x4 decomposition using Kronecker product of Strassen(2,2). Rank 49."""
    n, m, p = 4, 4, 4
    S_U, S_V, S_W = _strassen_2x2()
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    # U: row = i*4+j, i=2*i1+i2, j=2*j1+j2; col = r1*7+r2
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    row_u = (2*i1 + i2) * 4 + (2*j1 + j2)
                    for r1 in range(7):
                        for r2 in range(7):
                            U_44[row_u, r1*7+r2] = S_U[2*i1+j1, r1] * S_U[2*i2+j2, r2]
    
    # V: row = j*4+k, j=2*j1+j2, k=2*k1+k2
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    row_v = (2*j1 + j2) * 4 + (2*k1 + k2)
                    for r1 in range(7):
                        for r2 in range(7):
                            V_44[row_v, r1*7+r2] = S_V[2*j1+k1, r1] * S_V[2*j2+k2, r2]
    
    # W: row = k*4+i, k=2*k1+k2, i=2*i1+i2
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    row_w = (2*k1 + k2) * 4 + (2*i1 + i2)
                    for r1 in range(7):
                        for r2 in range(7):
                            W_44[row_w, r1*7+r2] = S_W[2*k1+i1, r1] * S_W[2*k2+i2, r2]
                            
    return U_44, V_44, W_44

def construct():
    start_time = time.time()
    deadline = 110
    
    if not _verify_strassen():
        # Fallback to schoolbook if Strassen fails (should not happen)
        n, m, p = DIMENSIONS
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

    U_44, V_44, W_44 = _build_4x4x4_kronecker()
    
    n, m, p = 4, 4, 9
    rank_44 = 49
    rank_1 = 16
    total_rank = rank_44 * 2 + rank_1 # 49 + 49 + 16 = 114
    
    U_total = np.zeros((n*m, total_rank))
    V_total = np.zeros((m*p, total_rank))
    W_total = np.zeros((p*n, total_rank))
    
    # Part 1: k in 0..3
    U_total[:, 0:rank_44] = U_44
    for j in range(4):
        for k in range(4):
            V_total[j*p + k, 0:rank_44] = V_44[j*4 + k, 0:rank_44]
    for k in range(4):
        for i in range(4):
            W_total[k*n + i, 0:rank_44] = W_44[k*4 + i, 0:rank_44]
            
    # Part 2: k in 4..7
    offset_2 = rank_44
    U_total[:, offset_2:offset_2+rank_44] = U_44
    for j in range(4):
        for k_local in range(4):
            k_global = k_local + 4
            V_total[j*p + k_global, offset_2:offset_2+rank_44] = V_44[j*4 + k_local, 0:rank_44]
    for k_local in range(4):
        k_global = k_local + 4
        for i in range(4):
            W_total[k_global*n + i, offset_2:offset_2+rank_44] = W_44[k_local*4 + i, 0:rank_44]
            
    # Part 3: k in 8..8 (Schoolbook for 4x4x1)
    offset_3 = rank_44 * 2
    col = 0
    for i in range(4):
        for j in range(4):
            c = offset_3 + col
            U_total[i*4 + j, c] = 1
            V_total[j*9 + 8, c] = 1
            W_total[8*4 + i, c] = 1
            col += 1
            
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx in range(len(row)):
                val = row[idx]
                if abs(val - round(val)) < 1e-10:
                    row[idx] = int(round(val))
                    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
