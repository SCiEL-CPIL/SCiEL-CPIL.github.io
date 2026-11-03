import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _strassen_2x2():
    """
    Returns U, V, W factors for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    # Strassen variables
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Row indices for A (U): 0:00, 1:01, 2:10, 3:11
    # Row indices for B (V): 0:00, 1:01, 2:10, 3:11
    # Row indices for C (W): 0:00, 1:01, 2:10, 3:11
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = -1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1; W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def _construct_decomposition():
    n, m, p = DIMENSIONS # 3, 4, 5
    
    # We build the decomposition as a list of (U_block, V_block, W_block, i_offset, j_offset, k_offset, i_size, j_size, k_size)
    # But it's easier to just append to the final matrices.
    
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # 1. Tensor 1: A1 (2x4) * B1 (4x4) -> C11 (2x4)
    # Split A1 into A1a (2x2), A1b (2x2)
    # Split B1 into B11 (2x2), B12 (2x2), B21 (2x2), B22 (2x2)
    # Terms:
    # T11: A1a * B11 (2x2x2) -> i in {0,1}, j in {0,1}, k in {0,1}
    # T12: A1a * B12 (2x2x2) -> i in {0,1}, j in {2,3}, k in {0,1}
    # T13: A1b * B21 (2x2x2) -> i in {0,1}, j in {0,1}, k in {2,3}
    # T14: A1b * B22 (2x2x2) -> i in {0,1}, j in {2,3}, k in {2,3}
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal total_rank
        U_s, V_s, W_s = _strassen_2x2()
        # Map local indices to global
        # Local U index: li * 2 + lj. Global: (i_off + li) * 4 + (j_off + lj)
        # Local V index: lj * 2 + lk. Global: (j_off + lj) * 5 + (k_off + lk)
        # Local W index: lk * 2 + li. Global: (k_off + lk) * 3 + (i_off + li)
        
        U_g = np.zeros((12, 7), dtype=int)
        V_g = np.zeros((20, 7), dtype=int)
        W_g = np.zeros((15, 7), dtype=int)
        
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    u_val = U_s[li*2+lj, r]
                    if u_val != 0:
                        gi = i_off + li
                        gj = j_off + lj
                        U_g[gi*4 + gj, r] = u_val
                        
            for lj in range(2):
                for lk in range(2):
                    v_val = V_s[lj*2+lk, r]
                    if v_val != 0:
                        gj = j_off + lj
                        gk = k_off + lk
                        V_g[gj*5 + gk, r] = v_val
                        
            for lk in range(2):
                for li in range(2):
                    w_val = W_s[lk*2+li, r]
                    if w_val != 0:
                        gk = k_off + lk
                        gi = i_off + li
                        W_g[gk*3 + gi, r] = w_val
                        
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        total_rank += 7
        
    add_strassen(0, 0, 0) # T11
    add_strassen(0, 2, 0) # T12
    add_strassen(0, 0, 2) # T13
    add_strassen(0, 2, 2) # T14
    
    # 2. Tensor 2: A1 (2x4) * B2 (4x1) -> C12 (2x1)
    # Split A1 into A1a (2x2), A1b (2x2)
    # Split B2 into B2a (2x1), B2b (2x1)
    # Terms:
    # T21: A1a * B2a (2x2x1) -> i in {0,1}, j in {0,1}, k in {4}
    # T22: A1b * B2b (2x2x1) -> i in {0,1}, j in {2,3}, k in {4}
    
    def add_schoolbook_221(i_off, j_off, k_off):
        nonlocal total_rank
        # Rank 4
        U_g = np.zeros((12, 4), dtype=int)
        V_g = np.zeros((20, 4), dtype=int)
        W_g = np.zeros((15, 4), dtype=int)
        
        col = 0
        for li in range(2):
            for lj in range(2):
                gi = i_off + li
                gj = j_off + lj
                gk = k_off
                
                U_g[gi*4 + gj, col] = 1
                V_g[gj*5 + gk, col] = 1
                W_g[gk*3 + gi, col] = 1
                col += 1
                
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        total_rank += 4
        
    add_schoolbook_221(0, 0, 4) # T21
    add_schoolbook_221(0, 2, 4) # T22
    
    # 3. Tensor 3: A2 (1x4) * B1 (4x4) -> C21 (1x4)
    # Schoolbook 1x4x4, Rank 16
    # i in {2}, j in {0,1,2,3}, k in {0,1,2,3}
    
    def add_schoolbook_144(i_off, j_off, k_off):
        nonlocal total_rank
        rank = 16
        U_g = np.zeros((12, rank), dtype=int)
        V_g = np.zeros((20, rank), dtype=int)
        W_g = np.zeros((15, rank), dtype=int)
        
        col = 0
        for lj in range(4):
            for lk in range(4):
                gi = i_off
                gj = j_off + lj
                gk = k_off + lk
                
                U_g[gi*4 + gj, col] = 1
                V_g[gj*5 + gk, col] = 1
                W_g[gk*3 + gi, col] = 1
                col += 1
                
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        total_rank += rank
        
    add_schoolbook_144(2, 0, 0) # T3
    
    # 4. Tensor 4: A2 (1x4) * B2 (4x1) -> C22 (1x1)
    # Schoolbook 1x4x1, Rank 4
    # i in {2}, j in {0,1,2,3}, k in {4}
    
    def add_schoolbook_141(i_off, j_off, k_off):
        nonlocal total_rank
        rank = 4
        U_g = np.zeros((12, rank), dtype=int)
        V_g = np.zeros((20, rank), dtype=int)
        W_g = np.zeros((15, rank), dtype=int)
        
        col = 0
        for lj in range(4):
            gi = i_off
            gj = j_off + lj
            gk = k_off
            
            U_g[gi*4 + gj, col] = 1
            V_g[gj*5 + gk, col] = 1
            W_g[gk*3 + gi, col] = 1
            col += 1
            
        U_list.append(U_g)
        V_list.append(V_g)
        W_list.append(W_g)
        total_rank += rank
        
    add_schoolbook_141(2, 0, 4) # T4
    
    U_final = np.hstack(U_list)
    V_final = np.hstack(V_list)
    W_final = np.hstack(W_list)
    
    return {"U": U_final, "V": V_final, "W": W_final}

def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_decomp = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_decomp[i,j,k] = s
    
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T_decomp, T_target, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    result = _construct_decomposition()
    
    if _verify(result['U'], result['V'], result['W']):
        return result
    
    # Fallback to schoolbook
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
