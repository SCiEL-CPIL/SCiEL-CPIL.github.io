import time
import numpy as np

DIMENSIONS = [3, 3, 8]
_START_TIME = time.time()
DEADLINE = _START_TIME + 110.0

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if n * m > 0 else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=tol)

def _strassen_222():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _outer_212():
    # C[i,j] = A[i] * B[j]
    # A is 2x1, B is 1x2, C is 2x2
    U = np.zeros((2, 4))
    V = np.zeros((2, 4))
    W = np.zeros((4, 4))
    
    # T0: A0*B0 -> C00
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # T1: A0*B1 -> C01
    U[0, 1] = 1
    V[1, 1] = 1
    W[1, 1] = 1
    
    # T2: A1*B0 -> C10
    U[1, 2] = 1
    V[0, 2] = 1
    W[2, 2] = 1
    
    # T3: A1*B1 -> C11
    U[1, 3] = 1
    V[1, 3] = 1
    W[3, 3] = 1
    
    return U, V, W

def _matvec_122():
    # C[0] = A[0]*B[0] + A[1]*B[1]
    # C[1] = A[0]*B[2] + A[1]*B[3]
    # A is 1x2, B is 2x2, C is 1x2
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    
    # T0: A0*B0 -> C0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # T1: A1*B1 -> C0
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # T2: A0*B2 -> C1
    U[0, 2] = 1
    V[2, 2] = 1
    W[1, 2] = 1
    
    # T3: A1*B3 -> C1
    U[1, 3] = 1
    V[3, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _scalarvec_112():
    # C[0] = A[0]*B[0]
    # C[1] = A[0]*B[1]
    # A is 1x1, B is 1x2, C is 1x2
    U = np.zeros((1, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))
    
    # T0: A0*B0 -> C0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # T1: A0*B1 -> C1
    U[0, 1] = 1
    V[1, 1] = 1
    W[1, 1] = 1
    
    return U, V, W

def _block_decompose():
    n, m, p = 3, 3, 8
    total_rank = 68
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    sU, sV, sW = _strassen_222()
    oU, oV, oW = _outer_212()
    mU, mV, mW = _matvec_122()
    cU, cV, cW = _scalarvec_112()
    
    col = 0
    for k in range(4):
        col_start = k * 2
        
        # 1) A00 * Btop_k (2x2 * 2x2) Rank 7
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = i * 3 + j
                    v_idx = i * 8 + (col_start + j)
                    w_idx = (col_start + j) * 3 + i
                    U[u_idx, col + r] += sU[i*2+j, r]
                    V[v_idx, col + r] += sV[i*2+j, r]
                    W[w_idx, col + r] += sW[i*2+j, r]
        col += 7
        
        # 2) A01 * Bbot_k (2x1 * 1x2) Rank 4
        for r in range(4):
            for i in range(2):
                for j in range(2):
                    u_idx = i * 3 + 2
                    v_idx = 2 * 8 + (col_start + j)
                    w_idx = (col_start + j) * 3 + i
                    U[u_idx, col + r] += oU[i, r]
                    V[v_idx, col + r] += oV[j, r]
                    W[w_idx, col + r] += oW[i*2+j, r]
        col += 4
        
        # 3) A10 * Btop_k (1x2 * 2x2) Rank 4
        for r in range(4):
            for j in range(2):
                for i in range(2):
                    # A10 is 1x2, index j. Btop is 2x2, index i, col_start+j?
                    # Wait, Btop_k is 2x2. Rows 0,1. Cols 0,1 (relative).
                    # A10[j] * Btop[i, j_rel]
                    # In our matvec_122, B is 2x2.
                    # V index corresponds to B element.
                    # Btop_k[i, j_rel] is at global B index i*8 + (col_start + j_rel).
                    # W index corresponds to C element.
                    # C is 1x2. C[j_rel].
                    # Global C index (col_start + j_rel)*3 + 2.
                    
                    u_idx = 2 * 3 + j
                    v_idx = i * 8 + (col_start + j_rel) # j_rel is the column in Btop
                    # In matvec_122, V is 4x4. V[row, col].
                    # row corresponds to B element index.
                    # B elements are 0,1 (col 0), 2,3 (col 1).
                    # So v_idx in local V is i*2 + j_rel.
                    # We need to map local V index to global V index.
                    
                    # Let's re-verify the mapping.
                    # mV is 4x4. mV[local_b_idx, r].
                    # local_b_idx = i*2 + j_rel.
                    # global_b_idx = i*8 + (col_start + j_rel).
                    
                    # mW is 2x4. mW[local_c_idx, r].
                    # local_c_idx = j_rel.
                    # global_c_idx = (col_start + j_rel)*3 + 2.
                    
                    # mU is 2x4. mU[local_a_idx, r].
                    # local_a_idx = j.
                    # global_a_idx = 2*3 + j.
                    
                    pass
        
        # Correct implementation for part 3
        for r in range(4):
            # U
            for j in range(2):
                u_idx = 2 * 3 + j
                U[u_idx, col + r] += mU[j, r]
            
            # V
            for i in range(2):
                for j_rel in range(2):
                    local_b_idx = i * 2 + j_rel
                    v_idx = i * 8 + (col_start + j_rel)
                    V[v_idx, col + r] += mV[local_b_idx, r]
            
            # W
            for j_rel in range(2):
                local_c_idx = j_rel
                w_idx = (col_start + j_rel) * 3 + 2
                W[w_idx, col + r] += mW[local_c_idx, r]
        
        col += 4
        
        # 4) A11 * Bbot_k (1x1 * 1x2) Rank 2
        for r in range(2):
            u_idx = 2 * 3 + 2
            U[u_idx, col + r] += cU[0, r]
            
            for j_rel in range(2):
                v_idx = 2 * 8 + (col_start + j_rel)
                V[v_idx, col + r] += cV[j_rel, r]
                
                w_idx = (col_start + j_rel) * 3 + 2
                W[w_idx, col + r] += cW[j_rel, r]
        
        col += 2
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

def _schoolbook():
    n, m, p = 3, 3, 8
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    result = _block_decompose()
    if _verify(result['U'], result['V'], result['W'], n, m, p):
        return result
    result = _schoolbook()
    if _verify(result['U'], result['V'], result['W'], n, m, p):
        return result
    return result
