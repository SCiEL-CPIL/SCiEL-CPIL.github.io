import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
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
        
        def add_strassen_term(start_col, a_blocks, b_blocks, c_blocks):
            for r in range(r3):
                u_vec = U3[:, r]
                v_vec = V3[:, r]
                w_vec = W3[:, r]
                u_mat = u_vec.reshape(3, 3)
                v_mat = v_vec.reshape(3, 3)
                w_mat = w_vec.reshape(3, 3)
                
                for bi, coeff in a_blocks:
                    row_block = bi // 2
                    col_block = bi % 2
                    for i in range(3):
                        for j in range(3):
                            val = coeff * u_mat[i, j]
                            if val != 0:
                                global_i = row_block * 3 + i
                                global_j = col_block * 3 + j
                                global_u_idx = global_i * 6 + global_j
                                U6[global_u_idx, start_col + r] += val
                
                for bi, coeff in b_blocks:
                    row_block = bi // 2
                    col_block = bi % 2
                    for j in range(3):
                        for k in range(3):
                            val = coeff * v_mat[j, k]
                            if val != 0:
                                global_j = row_block * 3 + j
                                global_k = col_block * 3 + k
                                global_v_idx = global_j * 6 + global_k
                                V6[global_v_idx, start_col + r] += val
                
                for bi, coeff in c_blocks:
                    row_block = bi // 2
                    col_block = bi % 2
                    for k in range(3):
                        for i in range(3):
                            val = coeff * w_mat[k, i]
                            if val != 0:
                                global_k = col_block * 3 + k
                                global_i = row_block * 3 + i
                                global_w_idx = global_k * 6 + global_i
                                W6[global_w_idx, start_col + r] += val

        add_strassen_term(0, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
        add_strassen_term(r3, [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])
        add_strassen_term(2*r3, [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])
        add_strassen_term(3*r3, [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])
        add_strassen_term(4*r3, [(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])
        add_strassen_term(5*r3, [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
        add_strassen_term(6*r3, [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
        
        return U6, V6, W6

    def verify_decomposition(U, V, W):
        U_reshaped = U.reshape(6, 6, -1)
        V_reshaped = V.reshape(6, 6, -1)
        W_reshaped = W.reshape(6, 6, -1)
        try:
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            target = np.ones((6, 6, 6))
            return np.allclose(reconstructed, target, atol=1e-9)
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

    U189, V189, W189 = construct_strassen_189()
    valid_189 = verify_decomposition(U189, V189, W189)
    
    if valid_189:
        return {"U": U189.tolist(), "V": V189.tolist(), "W": W189.tolist()}
    else:
        U, V, W = get_schoolbook_6x6()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
