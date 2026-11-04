import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 3, 4, 5
    
    def add_term(U_dict, V_dict, W_dict):
        U_vec = np.zeros(n*m)
        V_vec = np.zeros(m*p)
        W_vec = np.zeros(p*n)
        for (i,j) in U_dict:
            U_vec[i*m + j] = U_dict[(i,j)]
        for (j,k) in V_dict:
            V_vec[j*p + k] = V_dict[(j,k)]
        for (i,k) in W_dict:
            W_vec[k*n + i] = W_dict[(i,k)]
        return U_vec, V_vec, W_vec
    
    def strassen_2x2x2(i_off, j_off, k_off):
        cols = []
        # M1
        U1 = {(i_off, j_off): 1, (i_off+1, j_off+1): 1}
        V1 = {(j_off, k_off): 1, (j_off+1, k_off+1): 1}
        W1 = {(i_off, k_off): 1, (i_off+1, k_off+1): 1}
        cols.append(add_term(U1, V1, W1))
        # M2
        U2 = {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1}
        V2 = {(j_off, k_off): 1}
        W2 = {(i_off+1, k_off): 1}
        cols.append(add_term(U2, V2, W2))
        # M3
        U3 = {(i_off, j_off): 1}
        V3 = {(j_off+1, k_off): 1, (j_off+1, k_off+1): -1}
        W3 = {(i_off, k_off): 1, (i_off, k_off+1): -1}
        cols.append(add_term(U3, V3, W3))
        # M4
        U4 = {(i_off+1, j_off+1): 1}
        V4 = {(j_off, k_off): 1, (j_off, k_off+1): -1}
        W4 = {(i_off+1, k_off): 1, (i_off+1, k_off+1): -1}
        cols.append(add_term(U4, V4, W4))
        # M5
        U5 = {(i_off, j_off): 1, (i_off, j_off+1): 1}
        V5 = {(j_off+1, k_off+1): 1}
        W5 = {(i_off, k_off+1): 1}
        cols.append(add_term(U5, V5, W5))
        # M6
        U6 = {(i_off+1, j_off): 1, (i_off, j_off): -1}
        V6 = {(j_off, k_off): 1, (j_off, k_off+1): 1}
        W6 = {(i_off+1, k_off): 1, (i_off+1, k_off+1): 1, (i_off, k_off): -1, (i_off, k_off+1): -1}
        cols.append(add_term(U6, V6, W6))
        # M7
        U7 = {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1}
        V7 = {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1}
        W7 = {(i_off, k_off): 1, (i_off, k_off+1): 1, (i_off+1, k_off): -1, (i_off+1, k_off+1): -1}
        cols.append(add_term(U7, V7, W7))
        return cols

    def schoolbook_2x2x1(i_off, j_off, k_idx):
        # Correct schoolbook for 2x2x1
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Terms: A[i, j] B[j, k]
        cols = []
        for i_local in range(2):
            for j_local in range(2):
                U_dict = {(i_off + i_local, j_off + j_local): 1}
                V_dict = {(j_off + j_local, k_idx): 1}
                W_dict = {(i_off + i_local, k_idx): 1}
                cols.append(add_term(U_dict, V_dict, W_dict))
        return cols

    def schoolbook_1x2xp(i_idx, j_off, k_range):
        # Correct schoolbook for 1x2xp
        cols = []
        for j_local in range(2):
            for k in k_range:
                U_dict = {(i_idx, j_off + j_local): 1}
                V_dict = {(j_off + j_local, k): 1}
                W_dict = {(i_idx, k): 1}
                cols.append(add_term(U_dict, V_dict, W_dict))
        return cols

    columns = []
    
    # 2x4x4 part
    columns.extend(strassen_2x2x2(0, 0, 0))
    columns.extend(strassen_2x2x2(0, 0, 2))
    columns.extend(strassen_2x2x2(0, 2, 0))
    columns.extend(strassen_2x2x2(0, 2, 2))
    
    # 2x4x1 part
    columns.extend(schoolbook_2x2x1(0, 0, 4))
    columns.extend(schoolbook_2x2x1(0, 2, 4))
    
    # 1x4x5 part
    columns.extend(schoolbook_1x2xp(2, 0, range(5)))
    columns.extend(schoolbook_1x2xp(2, 2, range(5)))
    
    total_rank = len(columns)
    
    if time.time() > deadline:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U_final = np.zeros((n*m, total_rank))
    V_final = np.zeros((m*p, total_rank))
    W_final = np.zeros((p*n, total_rank))
    for c, (U_c, V_c, W_c) in enumerate(columns):
        U_final[:, c] = U_c
        V_final[:, c] = V_c
        W_final[:, c] = W_c
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_actual = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for c in range(total_rank):
                fa = np.dot(U_final[:, c], A_test.reshape(-1))
                fb = np.dot(V_final[:, c], B_test.reshape(-1))
                val += W_final[k*n + i, c] * fa * fb
            C_actual[i, k] = val
    
    is_valid = np.allclose(C_actual, C_expected, atol=1e-9)
    
    if not is_valid:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
