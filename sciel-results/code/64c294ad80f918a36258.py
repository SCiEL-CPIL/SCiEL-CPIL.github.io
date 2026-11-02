import numpy as np
import time

_START = time.time()

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    n = 4
    
    def get_strassen_base():
        # Returns U, V, W for 2x2 multiplication.
        # Indices:
        # A: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
        # B: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
        # C: (0,0), (0,1), (1,0), (1,1) -> W[k*2+i]
        # C[0,0] -> i=0, k=0 -> 0
        # C[0,1] -> i=0, k=1 -> 1*2+0 = 2
        # C[1,0] -> i=1, k=0 -> 0*2+1 = 1
        # C[1,1] -> i=1, k=1 -> 1*2+1 = 3
        
        U = np.zeros((4, 7), dtype=np.int64)
        V = np.zeros((4, 7), dtype=np.int64)
        W = np.zeros((4, 7), dtype=np.int64)
        
        def add(mat, idx, r, val):
            mat[idx, r] += val
            
        # P1 = (A00 + A11)(B00 + B11)
        add(U, 0, 0, 1); add(U, 3, 0, 1)
        add(V, 0, 0, 1); add(V, 3, 0, 1)
        add(W, 0, 0, 1); add(W, 3, 0, 1)
        
        # P2 = (A10 + A11)B00
        add(U, 2, 1, 1); add(U, 3, 1, 1)
        add(V, 0, 1, 1)
        add(W, 1, 1, 1)
        
        # P3 = A00(B01 - B11)
        add(U, 0, 2, 1)
        add(V, 1, 2, 1); add(V, 3, 2, -1)
        add(W, 2, 2, 1); add(W, 3, 2, -1)
        
        # P4 = A11(B10 - B00)
        add(U, 3, 3, 1)
        add(V, 2, 3, 1); add(V, 0, 3, -1)
        add(W, 0, 3, -1); add(W, 2, 3, 1)
        
        # P5 = (A00 + A01)B11
        add(U, 0, 4, 1); add(U, 1, 4, 1)
        add(V, 3, 4, 1)
        add(W, 1, 4, 1); add(W, 3, 4, 1)
        
        # P6 = (A10 - A00)(B00 + B01)
        add(U, 2, 5, 1); add(U, 0, 5, -1)
        add(V, 0, 5, 1); add(V, 1, 5, 1)
        add(W, 0, 5, 1); add(W, 3, 5, 1)
        
        # P7 = (A01 - A11)(B10 + B11)
        add(U, 1, 6, 1); add(U, 3, 6, -1)
        add(V, 2, 6, 1); add(V, 3, 6, 1)
        add(W, 0, 6, 1); add(W, 3, 6, 1)
        
        return U, V, W

    def build_strassen(n):
        if n == 2:
            return get_strassen_base()
        
        half = n // 2
        U_sub, V_sub, W_sub = build_strassen(half)
        rank_sub = U_sub.shape[1]
        rank = rank_sub * rank_sub
        
        U = np.zeros((n*n, rank), dtype=np.int64)
        V = np.zeros((n*n, rank), dtype=np.int64)
        W = np.zeros((n*n, rank), dtype=np.int64)
        
        terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): -1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (0,1): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (1,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ]
        
        for t_idx, (A_coeffs, B_coeffs, C_coeffs) in enumerate(terms):
            for r_sub in range(rank_sub):
                r_global = t_idx * rank_sub + r_sub
                
                for (br_a, bc_a), coeff_a in A_coeffs.items():
                    for lr in range(half):
                        for lc in range(half):
                            u_sub_idx = lr * half + lc
                            u_val = U_sub[u_sub_idx, r_sub]
                            if u_val != 0:
                                global_i = br_a * half + lr
                                global_j = bc_a * half + lc
                                u_idx = global_i * n + global_j
                                U[u_idx, r_global] += coeff_a * u_val
                                
                for (br_b, bc_b), coeff_b in B_coeffs.items():
                    for lr in range(half):
                        for lc in range(half):
                            v_sub_idx = lr * half + lc
                            v_val = V_sub[v_sub_idx, r_sub]
                            if v_val != 0:
                                global_j = br_b * half + lr
                                global_k = bc_b * half + lc
                                v_idx = global_j * n + global_k
                                V[v_idx, r_global] += coeff_b * v_val
                                
                for (br_c, bc_c), coeff_c in C_coeffs.items():
                    for lr in range(half):
                        for lc in range(half):
                            w_sub_idx = lr * half + lc
                            w_val = W_sub[w_sub_idx, r_sub]
                            if w_val != 0:
                                global_i = br_c * half + lr
                                global_k = bc_c * half + lc
                                w_idx = global_k * n + global_i
                                W[w_idx, r_global] += coeff_c * w_val

        return U, V, W

    U, V, W = build_strassen(n)
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
