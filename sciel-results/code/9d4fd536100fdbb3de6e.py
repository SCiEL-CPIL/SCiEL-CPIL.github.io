import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Laderman's 3x3 matrix multiplication algorithm (rank 23)
    # Patterns: (U_coeffs, V_coeffs, W_coeffs)
    # U is 3x3, V is 3x3, W is 3x3
    laderman_23_patterns = [
        ({(0,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1}),
        ({(1,0):1}, {(1,0):1,(2,0):1}, {(1,0):1}),
        ({(2,0):1}, {(2,0):1}, {(2,0):1}),
        ({(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1,(2,1):1}),
        ({(1,1):1}, {(2,1):1}, {(1,1):-1}),
        ({(2,1):1}, {(2,1):1}, {(2,1):-1}),
        ({(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1,(2,2):1}),
        ({(1,2):1}, {(2,2):1}, {(1,2):-1}),
        ({(2,2):1}, {(2,2):1}, {(2,2):-1}),
        ({(0,1):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,1):1}),
        ({(0,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,2):1}),
        ({(1,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1}),
        ({(1,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1}),
        ({(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(2,1):1}),
        ({(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(2,2):1}),
        ({(0,0):1}, {(1,1):1,(2,1):1}, {(0,1):1}),
        ({(0,0):1}, {(1,2):1,(2,2):1}, {(0,2):1}),
        ({(1,1):1}, {(2,2):1}, {(1,2):1}),
        ({(1,2):1}, {(2,1):1}, {(1,1):1}),
        ({(2,1):1}, {(2,2):1}, {(2,2):1}),
        ({(2,2):1}, {(2,1):1}, {(2,1):1}),
        ({(1,1):1,(2,2):1}, {(0,1):1,(1,2):1}, {(1,1):-1,(2,2):-1}),
        ({(1,2):1,(2,1):1}, {(0,2):1,(1,1):1}, {(1,2):-1,(2,1):-1}),
    ]

    # Strassen 2x2 patterns
    # U, V, W are 2x2
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def build_kronecker_6x6():
        """
        Build 6x6x6 decomposition using Kronecker product of Strassen (2x2, rank 7) 
        and Laderman (3x3, rank 23). Total rank = 7 * 23 = 161.
        """
        rank_strassen = 7
        rank_laderman = 23
        total_rank = rank_strassen * rank_laderman
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # Iterate over Strassen steps
        for s_idx, (u_str, v_str, w_str) in enumerate(strassen_patterns):
            # Iterate over Laderman steps
            for l_idx, (u_lad, v_lad, w_lad) in enumerate(laderman_23_patterns):
                r_idx = s_idx * rank_laderman + l_idx
                
                # Fill U
                for (i1, j1), val_str in u_str.items():
                    for (i2, j2), val_lad in u_lad.items():
                        i_global = i1 * 3 + i2
                        j_global = j1 * 3 + j2
                        U[i_global * m + j_global, r_idx] += val_str * val_lad
                        
                # Fill V
                for (j1, k1), val_str in v_str.items():
                    for (j2, k2), val_lad in v_lad.items():
                        j_global = j1 * 3 + j2
                        k_global = k1 * 3 + k2
                        V[j_global * p + k_global, r_idx] += val_str * val_lad
                        
                # Fill W
                for (i1, k1), val_str in w_str.items():
                    for (i2, k2), val_lad in w_lad.items():
                        i_global = i1 * 3 + i2
                        k_global = k1 * 3 + k2
                        # W index is k*n + i
                        W[k_global * n + i_global, r_idx] += val_str * val_lad
                        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank

    # Test matrices for verification
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    def verify_decomposition(U_arr, V_arr, W_arr, A_test, B_test, C_expected):
        rank_used = U_arr.shape[1]
        C_computed = np.zeros((n, p))
        # Vectorized verification
        # C_computed = sum_r (U[:, r] * A_flat) * (V[:, r] * B_flat) * W[:, r]
        # A_flat is n*m, U is n*m x rank
        # M_u = U^T A_flat (rank x 1)
        # M_v = V^T B_flat (rank x 1)
        # M_r = M_u * M_v (rank x 1)
        # C_flat = W^T M_r (n*p x 1)
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        M_u = np.dot(U_arr.T, A_flat)
        M_v = np.dot(V_arr.T, B_flat)
        M_r = M_u * M_v
        C_flat = np.dot(W_arr.T, M_r)
        
        C_computed = C_flat.reshape(p, n).T
        error = np.max(np.abs(C_computed - C_expected))
        return error

    # Build the Kronecker solution
    sol, rank = build_kronecker_6x6()
    U_l = np.array(sol["U"], dtype=float)
    V_l = np.array(sol["V"], dtype=float)
    W_l = np.array(sol["W"], dtype=float)
    
    err = verify_decomposition(U_l, V_l, W_l, A_test, B_test, C_expected)
    
    if err < 1e-10:
        return sol
    else:
        # Fallback to schoolbook if something went wrong (should not happen)
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, idx] = 1
                    V_s[j * p + k, idx] = 1
                    W_s[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
