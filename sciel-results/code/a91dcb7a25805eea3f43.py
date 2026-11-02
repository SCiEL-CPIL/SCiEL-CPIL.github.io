import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 8
    
    def strassen_2x2():
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[1, 5] = 1; W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[2, 6] = 1
        
        return U, V, W

    def strassen_4x4():
        U_2, V_2, W_2 = strassen_2x2()
        
        rank = 49
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        
        a_blocks = [
            [(0, 1), (3, 1)],
            [(2, 1), (3, 1)],
            [(0, 1)],
            [(3, 1)],
            [(0, 1), (1, 1)],
            [(2, 1), (0, -1)],
            [(1, 1), (3, -1)],
        ]
        
        b_blocks = [
            [(0, 1), (3, 1)],
            [(0, 1)],
            [(1, 1), (3, -1)],
            [(2, 1), (0, -1)],
            [(3, 1)],
            [(0, 1), (1, 1)],
            [(2, 1), (3, 1)],
        ]
        
        c_blocks = [
            [(0, 1), (3, 1)],
            [(2, 1), (3, -1)],
            [(1, 1), (3, 1)],
            [(0, 1), (2, 1)],
            [(0, -1), (1, 1)],
            [(2, 1), (3, 1)],
            [(0, -1), (1, 1)],
        ]
        
        def block_row(b):
            return (b // 2) * 2
        
        def block_col(b):
            return (b % 2) * 2
        
        col = 0
        for j in range(7):
            for t in range(7):
                for a_idx, a_coeff in a_blocks[j]:
                    ar = block_row(a_idx)
                    ac = block_col(a_idx)
                    for i2 in range(2):
                        for j2 in range(2):
                            val = a_coeff * U_2[i2*2+j2, t]
                            if val != 0:
                                U[(ar + i2)*4 + (ac + j2), col] += val
                
                for b_idx, b_coeff in b_blocks[j]:
                    br = block_row(b_idx)
                    bc = block_col(b_idx)
                    for j2 in range(2):
                        for k2 in range(2):
                            val = b_coeff * V_2[j2*2+k2, t]
                            if val != 0:
                                V[(br + j2)*4 + (bc + k2), col] += val
                
                for c_idx, c_coeff in c_blocks[j]:
                    cr = block_row(c_idx)
                    cc = block_col(c_idx)
                    for k2 in range(2):
                        for i2 in range(2):
                            val = c_coeff * W_2[k2*2+i2, t]
                            if val != 0:
                                W[(cr + k2)*4 + (cc + i2), col] += val
                
                col += 1
        
        return U, V, W

    U_4, V_4, W_4 = strassen_4x4()
    
    # Verify 4x4x4
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1.0
                actual = 0.0
                for r in range(49):
                    actual += U_4[i*4+j, r] * V_4[j*4+k, r] * W_4[k*4+i, r]
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err

    if max_err > 1e-10:
        # Fallback to schoolbook 4x4x4
        rank = 64
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        # But we need 4x4x8
        # So we just return schoolbook 4x4x8 directly
        rank = 128
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((32, rank), dtype=float)
        W = np.zeros((32, rank), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    U[i*4+j, col] = 1
                    V[j*8+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Construct 4x4x8
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # First 4x4x4 (k=0..3)
    for r in range(49):
        U[:, r] = U_4[:, r]
        V[0:16, r] = V_4[:, r]
        W[0:16, r] = W_4[:, r]
        
    # Second 4x4x4 (k=4..7)
    for r in range(49):
        U[:, 49+r] = U_4[:, r]
        V[16:32, 49+r] = V_4[:, r]
        W[16:32, 49+r] = W_4[:, r]
        
    # Verify 4x4x8
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(8):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r]
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err

    if max_err > 1e-10:
        rank = 128
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((32, rank), dtype=float)
        W = np.zeros((32, rank), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    U[i*4+j, col] = 1
                    V[j*8+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
