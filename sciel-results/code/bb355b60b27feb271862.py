import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12
    
    # Strassen decomposition for 2x2x2
    # 7 terms
    strassen_U = np.zeros((2, 7))
    strassen_V = np.zeros((2, 7))
    strassen_W = np.zeros((2, 7))
    
    # M1 = (A00+A11)(B00+B11)
    strassen_U[0, 0] = 1; strassen_U[1, 0] = 1
    strassen_V[0, 0] = 1; strassen_V[1, 0] = 1
    strassen_W[0, 0] = 1; strassen_W[1, 0] = 1
    
    # M2 = (A10+A11)B00
    strassen_U[1, 1] = 1
    strassen_V[0, 1] = 1
    strassen_W[0, 1] = 1; strassen_W[1, 1] = -1
    
    # M3 = A00(B01-B11)
    strassen_U[0, 2] = 1
    strassen_V[1, 2] = 1; strassen_V[0, 2] = -1
    strassen_W[1, 2] = 1; strassen_W[0, 2] = -1
    
    # M4 = A11(B10-B00)
    strassen_U[1, 3] = 1
    strassen_V[1, 3] = 1; strassen_V[0, 3] = -1
    strassen_W[0, 3] = 1; strassen_W[1, 3] = -1
    
    # M5 = (A00+A01)B11
    strassen_U[0, 4] = 1; strassen_U[1, 4] = 1
    strassen_V[1, 4] = 1
    strassen_W[0, 4] = -1; strassen_W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    strassen_U[1, 5] = 1; strassen_U[0, 5] = -1
    strassen_V[0, 5] = 1; strassen_V[1, 5] = 1
    strassen_W[0, 5] = 1; strassen_W[1, 5] = -1
    
    # M7 = (A01-A11)(B10+B11)
    strassen_U[0, 6] = 1; strassen_U[1, 6] = -1
    strassen_V[1, 6] = 1; strassen_V[0, 6] = 1
    strassen_W[0, 6] = 1; strassen_W[1, 6] = -1
    
    # Expand to 2x4x6 using WGA: T_{2x4x6} = T_{1x2x3} ⊗ T_{2x2x2}
    # T_{1x2x3} is rank 1: u=[1], v=[1,1], w=[1,1,1]
    # So U_full[i, j1*2+j2] = strassen_U[i, t] * 1
    #    V_full[j1*2+j2, k1*2+k2] = strassen_V[j2, t] * 1
    #    W_full[k1*2+k2, i] = strassen_W[k2, t] * 1
    
    rank = 7
    U = np.zeros((nm, rank))  # (8, 7)
    V = np.zeros((mp, rank))  # (24, 7)
    W = np.zeros((pn, rank))  # (12, 7)
    
    for t in range(rank):
        # U: i in {0,1}, j = j1*2 + j2 where j1 in {0,1}, j2 in {0,1}
        for i in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    U[i, t] += strassen_U[i, t] * 1  # v[j1] = 1
                    # Wait, this is wrong. U should be indexed by (i, j) but in the tensor
                    # decomposition, U has shape (n*m, rank) = (8, rank)
                    # U[i*m + j, t]
        
        # Let me redo: U has shape (n*m, rank) = (8, 7)
        # U[i*m + j, t] where i in {0,1}, j in {0,1,2,3}
        # j = j1*2 + j2
        for i in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    U[i * m + j, t] = strassen_U[i, t]  # * v[j1] = 1
        
        # V has shape (m*p, rank) = (24, 7)
        # V[j*p + k, t] where j in {0,1,2,3}, k in {0,1,2,3,4,5}
        # j = j1*2 + j2, k = k1*2 + k2
        for j1 in range(2):
            for j2 in range(2):
                for k1 in range(3):
                    for k2 in range(2):
                        j = j1 * 2 + j2
                        k = k1 * 2 + k2
                        V[j * p + k, t] = strassen_V[j2, t]  # * w[k1] = 1
        
        # W has shape (p*n, rank) = (12, 7)
        # W[k*n + i, t] where k in {0,1,2,3,4,5}, i in {0,1}
        # k = k1*2 + k2
        for k1 in range(3):
            for k2 in range(2):
                for i in range(2):
                    k = k1 * 2 + k2
                    W[k * n + i, t] = strassen_W[k2, t]  # * u[i1] = 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    i_w = c % n
                    k_w = c // n
                    if i_a == i_w and j_a == j_b and k_b == k_w:
                        continue
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s) > tol:
                        return False
        return True
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 42 Strassen blocks
    rank_init = 42
    U = np.zeros((nm, rank_init), dtype=float)
    V = np.zeros((mp, rank_init), dtype=float)
    W = np.zeros((pn, rank_init), dtype=float)
    
    def add_strassen_block(U, V, W, i_start, j_start, k_start, offset):
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1
        
        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0
        
        U[a10, offset+1] += 1.0; U[a11, offset+1] += 1.0
        V[b00, offset+1] += 1.0
        W[w10, offset+1] += 1.0; W[w11, offset+1] -= 1.0
        
        U[a00, offset+2] += 1.0
        V[b01, offset+2] += 1.0; V[b11, offset+2] -= 1.0
        W[w01, offset+2] += 1.0; W[w11, offset+2] += 1.0
        
        U[a11, offset+3] += 1.0
        V[b10, offset+3] += 1.0; V[b00, offset+3] -= 1.0
        W[w00, offset+3] += 1.0; W[w10, offset+3] += 1.0
        
        U[a00, offset+4] += 1.0; U[a01, offset+4] += 1.0
        V[b11, offset+4] += 1.0
        W[w00, offset+4] -= 1.0; W[w01, offset+4] += 1.0
        
        U[a10, offset+5] += 1.0; U[a00, offset+5] -= 1.0
        V[b00, offset+5] += 1.0; V[b01, offset+5] += 1.0
        W[w11, offset+5] += 1.0
        
        U[a01, offset+6] += 1.0; U[a11, offset+6] -= 1.0
        V[b10, offset+6] += 1.0; V[b11, offset+6] += 1.0
        W[w00, offset+6] += 1.0
    
    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(U, V, W, i_start, j_start, k_start, offset)
                offset += 7
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
