import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit
    
    def get_schoolbook_3x3():
        n3 = 3
        rank3 = 27
        U3 = np.zeros((9, rank3))
        V3 = np.zeros((9, rank3))
        W3 = np.zeros((9, rank3))
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    col = i*9 + j*3 + k
                    U3[i*3+j, col] = 1.0
                    V3[j*3+k, col] = 1.0
                    W3[k*3+i, col] = 1.0
        return U3, V3, W3

    def construct_strassen_189():
        U3, V3, W3 = get_schoolbook_3x3()
        r3 = 27
        total_rank = 7 * r3
        U6 = np.zeros((36, total_rank))
        V6 = np.zeros((36, total_rank))
        W6 = np.zeros((36, total_rank))
        
        def block_row_col_to_global_idx(rb, cb, i, j):
            return (rb * 3 + i) * 6 + (cb * 3 + j)
        
        strassen_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        for term_idx, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_terms):
            start_col = term_idx * r3
            
            for r in range(r3):
                u_vec = U3[:, r]
                v_vec = V3[:, r]
                w_vec = W3[:, r]
                
                if time.time() > deadline - 5.0:
                    return None, None, None
                
                for bi, coeff in a_blocks:
                    rb = bi // 2
                    cb = bi % 2
                    for i in range(3):
                        for j in range(3):
                            val = coeff * u_vec[i*3+j]
                            if val != 0:
                                U6[block_row_col_to_global_idx(rb, cb, i, j), start_col + r] += val
                
                for bi, coeff in b_blocks:
                    rb = bi // 2
                    cb = bi % 2
                    for j in range(3):
                        for k in range(3):
                            val = coeff * v_vec[j*3+k]
                            if val != 0:
                                V6[block_row_col_to_global_idx(rb, cb, j, k), start_col + r] += val
                
                for bi, coeff in c_blocks:
                    rb = bi // 2
                    cb = bi % 2
                    for k in range(3):
                        for i in range(3):
                            val = coeff * w_vec[k*3+i]
                            if val != 0:
                                W6[(cb * 3 + k) * 6 + (rb * 3 + i), start_col + r] += val
        
        return U6, V6, W6

    def verify_decomposition(U, V, W):
        U_reshaped = U.reshape(6, 6, -1)
        V_reshaped = V.reshape(6, 6, -1)
        W_reshaped = W.reshape(6, 6, -1)
        try:
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            target = np.ones((6, 6, 6))
            return np.allclose(reconstructed, target, atol=1e-8)
        except:
            return False

    def get_schoolbook_6x6():
        rank = 216
        U = np.zeros((36, rank))
        V = np.zeros((36, rank))
        W = np.zeros((36, rank))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    idx = i*6 + j
                    jk = j*6 + k
                    ki = k*6 + i
                    col = i*36 + j*6 + k
                    U[idx, col] = 1.0
                    V[jk, col] = 1.0
                    W[ki, col] = 1.0
        return U, V, W

    # Try Strassen decomposition first
    U189, V189, W189 = construct_strassen_189()
    
    if U189 is not None:
        valid_189 = verify_decomposition(U189, V189, W189)
        if valid_189:
            return {"U": U189.tolist(), "V": V189.tolist(), "W": W189.tolist()}
    
    # Fallback to schoolbook
    U, V, W = get_schoolbook_6x6()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
