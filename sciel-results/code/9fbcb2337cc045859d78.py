import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_DEADLINE = time.time() + 140

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # k=0
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    # k=1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    # k=2
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    # k=3
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    # k=4
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = 1; W[0,4] = -1
    # k=5
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    # k=6
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _strassen_4x4x4():
    """Return U, V, W for 4x4x4 Strassen decomposition.
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # C blocks: C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
    
    # T0 = (A00+A11)(B00+B11)
    # T1 = (A10+A11)B00
    # T2 = A00(B01-B11)
    # T3 = A11(B10-B00)
    # T4 = (A00+A01)B11
    # T5 = (A10-A00)(B00+B01)
    # T6 = (A01-A11)(B10+B11)
    
    for k in range(7):
        col_offset = k * 7
        Uk = U2[:, k]
        Vk = V2[:, k]
        Wk = W2[:, k]
        
        if k == 0:
            a_idx = [0, 3]
            b_idx = [0, 3]
            c_idx = [0, 3]
        elif k == 1:
            a_idx = [2, 3]
            b_idx = [0]
            c_idx = [2, 3]
        elif k == 2:
            a_idx = [0]
            b_idx = [1, 3]
            c_idx = [1, 3]
        elif k == 3:
            a_idx = [3]
            b_idx = [2, 0]
            c_idx = [0, 2]
        elif k == 4:
            a_idx = [0, 1]
            b_idx = [3]
            c_idx = [1, 0]
        elif k == 5:
            a_idx = [2, 0]
            b_idx = [0, 1]
            c_idx = [3]
        elif k == 6:
            a_idx = [1, 3]
            b_idx = [2, 3]
            c_idx = [0]
            
        for i in range(len(a_idx)):
            a_block = a_idx[i]
            a_row = a_block * 4
            for r in range(4):
                U[a_row + r, col_offset + i] = Uk[r]
                
        for i in range(len(b_idx)):
            b_block = b_idx[i]
            b_row = b_block * 4
            for r in range(4):
                V[b_row + r, col_offset + i] = Vk[r]
                
        for i in range(len(c_idx)):
            c_block = c_idx[i]
            c_row = c_block * 4
            for r in range(4):
                W[c_row + r, col_offset + i] = Wk[r]
                
    return U, V, W

def _schoolbook_4x4x5():
    """Return U, V, W for 4x4x5 schoolbook decomposition.
    U: (16, 80)
    V: (20, 80)
    W: (20, 80)
    """
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Get components
    U_S, V_S, W_S = _strassen_4x4x4()
    U_sc, V_sc, W_sc = _schoolbook_4x4x5()
    
    rank_S = 49
    rank_sc = 80
    total_rank = rank_S + rank_sc
    
    # Initialize full matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Fill Strassen Part (first 49 columns) ---
    # U is same shape (16, 49)
    U[:, :rank_S] = U_S
    
    # V_S is (16, 49) corresponding to 4x4 B.
    # Global V is (36, 49) corresponding to 4x9 B.
    # Local index: j * 4 + k_loc (k_loc in 0..3)
    # Global index: j * 9 + k_glob (k_glob in 0..3)
    for j in range(4):
        for k_loc in range(4):
            local_row = j * 4 + k_loc
            global_row = j * 9 + k_loc
            V[global_row, :rank_S] = V_S[local_row, :rank_S]
    
    # W_S is (16, 49) corresponding to 4x4 C.
    # Global W is (36, 49) corresponding to 4x9 C.
    # Local index: k_loc * 4 + i (k_loc in 0..3)
    # Global index: k_glob * 4 + i (k_glob in 0..3)
    # Since k_glob == k_loc, we just copy the first 16 rows.
    W[:16, :rank_S] = W_S
    
    # --- Fill Schoolbook Part (last 80 columns) ---
    offset = rank_S
    # U is same shape (16, 80)
    U[:, offset:offset+rank_sc] = U_sc
    
    # V_sc is (20, 80) corresponding to 4x5 B.
    # Global V is (36, 80).
    # Local index: j * 5 + k_loc (k_loc in 0..4)
    # Global index: j * 9 + (4 + k_loc)
    for j in range(4):
        for k_loc in range(5):
            local_row = j * 5 + k_loc
            global_row = j * 9 + (4 + k_loc)
            V[global_row, offset:offset+rank_sc] = V_sc[local_row, :]
    
    # W_sc is (20, 80) corresponding to 4x5 C.
    # Global W is (36, 80).
    # Local index: k_loc * 4 + i (k_loc in 0..4)
    # Global index: (4 + k_loc) * 4 + i
    for k_loc in range(5):
        local_row = k_loc * 4
        global_row = (4 + k_loc) * 4
        W[global_row:global_row+4, offset:offset+rank_sc] = W_sc[local_row:local_row+4, :]
    
    # Verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to full schoolbook 4x4x9
        rank = n * m * p
        U_f = np.zeros((n * m, rank))
        V_f = np.zeros((m * p, rank))
        W_f = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
