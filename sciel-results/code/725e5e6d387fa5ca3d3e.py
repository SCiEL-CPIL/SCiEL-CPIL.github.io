"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Schoolbook rank is n*m*p = 96
    best_rank = n * m * p
    best_result = None
    
    def build_schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
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
    
    def validate_einsum(U, V, W, rank):
        """
        Validation using einsum for speed.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_r = U_arr.reshape(n, m, rank)
        V_r = V_arr.reshape(m, p, rank)
        W_r = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        return np.allclose(T, 1.0, atol=1e-8)

    def build_strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 multiplication using Strassen (rank 7).
        U: (4, 7), V: (4, 7), W: (4, 7)
        """
        U = np.zeros((4, 7), dtype=np.int8)
        V = np.zeros((4, 7), dtype=np.int8)
        W = np.zeros((4, 7), dtype=np.int8)
        
        # M1 = (A00+A11)(B00+B11)
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 1]
        W[:, 0] = [1, 0, 0, 1] # C00, C11
        
        # M2 = (A10+A11)B00
        U[:, 1] = [0, 0, 1, 1]
        V[:, 1] = [1, 0, 0, 0]
        W[:, 1] = [0, 1, 0, -1] # C10, C11
        
        # M3 = A00(B01-B11)
        U[:, 2] = [1, 0, 0, 0]
        V[:, 2] = [0, 1, 0, -1]
        W[:, 2] = [0, 0, 1, 1] # C01, C11
        
        # M4 = A11(B10-B00)
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [0, 0, 1, -1]
        W[:, 3] = [1, 1, 0, 0] # C00, C10
        
        # M5 = (A00+A01)B11
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1]
        W[:, 4] = [-1, 0, 1, 0] # C00, C01
        
        # M6 = (A10-A00)(B00+B01)
        U[:, 5] = [-1, 0, 1, 0]
        V[:, 5] = [1, 1, 0, 0]
        W[:, 5] = [0, 0, 0, 1] # C11
        
        # M7 = (A01-A11)(B10+B11)
        U[:, 6] = [0, 1, 0, -1]
        V[:, 6] = [0, 0, 1, 1]
        W[:, 6] = [1, 0, 0, 0] # C00
        
        return U, V, W

    def build_schoolbook_2x2x1():
        """
        Returns U, V, W for 2x2x1 multiplication (rank 4).
        U: (4, 4), V: (2, 4), W: (2, 4)
        """
        U = np.zeros((4, 4), dtype=np.int8)
        V = np.zeros((2, 4), dtype=np.int8)
        W = np.zeros((2, 4), dtype=np.int8)
        
        # C0 = A00*B0 + A01*B1
        U[:, 0] = [1, 0, 0, 0]
        V[:, 0] = [1, 0]
        W[:, 0] = [1, 0]
        
        U[:, 1] = [0, 1, 0, 0]
        V[:, 1] = [0, 1]
        W[:, 1] = [1, 0]
        
        # C1 = A10*B0 + A11*B1
        U[:, 2] = [0, 0, 1, 0]
        V[:, 2] = [1, 0]
        W[:, 2] = [0, 1]
        
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [0, 1]
        W[:, 3] = [0, 1]
        
        return U, V, W

    def build_optimized():
        """
        Constructs decomposition for 4x4x6 with rank 65.
        Split into 4x4x4 (rank 49) and 4x4x2 (rank 16).
        """
        U_strassen, V_strassen, W_strassen = build_strassen_2x2x2()
        U_sb, V_sb, W_sb = build_schoolbook_2x2x1()
        
        rank_total = 49 + 16
        U = np.zeros((16, rank_total), dtype=np.int16)
        V = np.zeros((24, rank_total), dtype=np.int16)
        W = np.zeros((24, rank_total), dtype=np.int16)
        
        # Part 1: 4x4x4 (k=0..3)
        # Outer Strassen indices s=0..6
        # Inner Strassen indices t=0..6
        for s in range(7):
            for t in range(7):
                c = s * 7 + t
                for i_b in range(2):
                    for j_b in range(2):
                        for k_b in range(2):
                            for i_l in range(2):
                                for j_l in range(2):
                                    for k_l in range(2):
                                        # Global indices
                                        i = 2 * i_b + i_l
                                        j = 2 * j_b + j_l
                                        k = 2 * k_b + k_l
                                        
                                        # Outer coefficients
                                        u_out = U_strassen[i_b * 2 + j_b, s]
                                        v_out = V_strassen[j_b * 2 + k_b, s]
                                        w_out = W_strassen[k_b * 2 + i_b, s]
                                        
                                        # Inner coefficients
                                        u_in = U_strassen[i_l * 2 + j_l, t]
                                        v_in = V_strassen[j_l * 2 + k_l, t]
                                        w_in = W_strassen[k_l * 2 + i_l, t]
                                        
                                        U[i, j, c] += u_out * u_in
                                        V[j, k, c] += v_out * v_in
                                        W[k, i, c] += w_out * w_in
                                        
        # Part 2: 4x4x2 (k=4..5)
        # Terms:
        # 1. A00 * B0k -> C0k (i_b=0, j_b=0, k'=0..1)
        # 2. A01 * B1k -> C0k (i_b=0, j_b=1, k'=0..1)
        # 3. A10 * B0k -> C1k (i_b=1, j_b=0, k'=0..1)
        # 4. A11 * B1k -> C1k (i_b=1, j_b=1, k'=0..1)
        
        base_rank = 49
        term_idx = 0
        for k_prime in range(2):
            for i_b in range(2):
                for j_b in range(2):
                    # This term corresponds to A[i_b, j_b] * B[j_b, k_prime] -> C[i_b, k_prime]
                    # Using schoolbook 2x2x1
                    for t in range(4):
                        c = base_rank + term_idx * 4 + t
                        
                        for i_l in range(2):
                            for j_l in range(2):
                                # Global i, j
                                i = 2 * i_b + i_l
                                j = 2 * j_b + j_l
                                
                                # Global k (column of B)
                                k = 4 + k_prime
                                
                                # Local indices for 2x2x1
                                # A is 2x2, B is 2x1, C is 2x1
                                # U_local[i_l, j_l, t]
                                # V_local[j_l, k_prime, t] (k_prime is the column index 0 or 1)
                                # W_local[k_prime, i_l, t]
                                
                                u_val = U_sb[i_l * 2 + j_l, t]
                                v_val = V_sb[j_l * 2 + k_prime, t]
                                w_val = W_sb[k_prime * 2 + i_l, t]
                                
                                U[i, j, c] += u_val
                                V[j, k, c] += v_val
                                W[k, i, c] += w_val
                                
                    term_idx += 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Start with schoolbook
    school = build_schoolbook()
    if validate_einsum(school["U"], school["V"], school["W"], best_rank):
        best_result = school
    
    # Try optimized
    if time.time() < deadline:
        opt = build_optimized()
        opt_rank = len(opt["U"][0])
        if validate_einsum(opt["U"], opt["V"], opt["W"], opt_rank):
            best_rank = opt_rank
            best_result = opt

    if best_result is None:
        best_result = school
    
    return best_result
# EVOLVE-BLOCK-END
