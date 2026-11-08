import numpy as np
import time

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin before 150s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Rank 56.
    """
    n, m, p = DIMENSIONS
    
    if time.time() > _DEADLINE:
        return _schoolbook_fallback(n, m, p)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W

    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W

    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W

    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            u = np.zeros(n * m)
            for i, idx in enumerate(u_idx):
                u[idx] = U_blk[i, r]
            U_parts.append(u)
            
            v = np.zeros(m * p)
            for i, idx in enumerate(v_idx):
                v[idx] = V_blk[i, r]
            V_parts.append(v)
            
            w = np.zeros(p * n)
            for i, idx in enumerate(w_idx):
                w[idx] = W_blk[i, r]
            W_parts.append(w)

    # Indices for U (3x4), V (4x5), W (5x3)
    # U[i*m + j]
    # V[j*p + k]
    # W[k*n + i]
    
    # Blocks for Rows 0,1 (2x4)
    # A1_top (2x2): U[0,1,4,5]
    # A1_bot (2x2): U[2,3,6,7]
    
    # B_top_left (2x2): V[0,1,5,6]
    # B_top_mid (2x2): V[2,3,7,8]
    # B_top_right (2x1): V[4,9]
    
    # B_bot_left (2x2): V[10,11,15,16]
    # B_bot_mid (2x2): V[12,13,17,18]
    # B_bot_right (2x1): V[14,19]
    
    # W indices for C (3x5)
    # C00 (W[0]), C01 (W[3]), C02 (W[6]), C03 (W[9]), C04 (W[12])
    # C10 (W[1]), C11 (W[4]), C12 (W[7]), C13 (W[10]), C14 (W[13])
    # C20 (W[2]), C21 (W[5]), C22 (W[8]), C23 (W[11]), C24 (W[14])
    
    # 1. A1_top * B_top_left (2x2x2)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([0,1,4,5], [0,1,5,6], [0,3,1,4], U_b, V_b, W_b)
    
    # 2. A1_top * B_top_mid (2x2x2)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([0,1,4,5], [2,3,7,8], [6,9,7,10], U_b, V_b, W_b)
    
    # 3. A1_top * B_top_right (2x2x1)
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block([0,1,4,5], [4,9], [12,13], U_b, V_b, W_b)
    
    # 4. A1_bot * B_bot_left (2x2x2)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([2,3,6,7], [10,11,15,16], [0,3,1,4], U_b, V_b, W_b)
    
    # 5. A1_bot * B_bot_mid (2x2x2)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([2,3,6,7], [12,13,17,18], [6,9,7,10], U_b, V_b, W_b)
    
    # 6. A1_bot * B_bot_right (2x2x1)
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block([2,3,6,7], [14,19], [12,13], U_b, V_b, W_b)
    
    # 7. A2_top * B_top_left (1x2x2)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block([8,9], [0,1,5,6], [2,5], U_b, V_b, W_b)
    
    # 8. A2_top * B_top_mid (1x2x2)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block([8,9], [2,3,7,8], [8,11], U_b, V_b, W_b)
    
    # 9. A2_top * B_top_right (1x2x1)
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_block([8,9], [4,9], [14], U_b, V_b, W_b)
    
    # 10. A2_bot * B_bot_left (1x2x2)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block([10,11], [10,11,15,16], [2,5], U_b, V_b, W_b)
    
    # 11. A2_bot * B_bot_mid (1x2x2)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block([10,11], [12,13,17,18], [8,11], U_b, V_b, W_b)
    
    # 12. A2_bot * B_bot_right (1x2x1)
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_block([10,11], [14,19], [14], U_b, V_b, W_b)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook_fallback(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    U_parts = []
    V_parts = []
    W_parts = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                u[i * m + j] = 1
                v = np.zeros(m * p)
                v[j * p + k] = 1
                w = np.zeros(p * n)
                w[k * n + i] = 1
                U_parts.append(u)
                V_parts.append(v)
                W_parts.append(w)
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
