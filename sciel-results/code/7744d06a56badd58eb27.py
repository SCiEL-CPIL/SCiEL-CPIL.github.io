import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock limit

    # Strassen 2x2 decomposition (Rank 7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: M1 = (A11+A22)(B11+B22) -> C11+, C22+
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2 = (A21+A22)B11 -> C21+, C22-
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=2: M3 = A11(B12-B22) -> C12+, C22+
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=3: M4 = A22(B21-B11) -> C11+, C21+
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=4: M5 = (A11+A12)B22 -> C11-, C12+
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5: M6 = (A21-A11)(B11+B12) -> C22+
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M7 = (A12-A22)(B21+B22) -> C11+
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Laderman 3x3 decomposition (Rank 23)
    U3 = np.zeros((9, 23), dtype=int)
    V3 = np.zeros((9, 23), dtype=int)
    W3 = np.zeros((9, 23), dtype=int)
    
    # Correct Laderman patterns
    # (A_coeffs, B_coeffs, C_coeffs)
    laderman_patterns = [
        ({(0,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1}),
        ({(1,0):1}, {(1,0):1,(2,0):1}, {(1,0):1}),
        ({(2,0):1}, {(2,0):1}, {(2,0):1}),
        ({(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,0):1,(2,0):1}),
        ({(1,1):1}, {(2,1):1}, {(1,0):1}),
        ({(2,1):1}, {(2,1):1}, {(2,0):1}),
        ({(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,0):1,(2,0):1}),
        ({(1,2):1}, {(2,2):1}, {(1,0):1}),
        ({(2,2):1}, {(2,2):1}, {(2,0):1}),
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
        ({(1,1):1,(2,2):1}, {(0,1):1,(1,2):1}, {(1,1):1,(2,2):1}),
        ({(1,2):1,(2,1):1}, {(0,2):1,(1,1):1}, {(1,2):1,(2,1):1}),
    ]
    
    for idx, (a_coeffs, b_coeffs, c_coeffs) in enumerate(laderman_patterns):
        for (pi, pj), val in a_coeffs.items():
            U3[pi * 3 + pj, idx] = val
        for (pi, pj), val in b_coeffs.items():
            V3[pi * 3 + pj, idx] = val
        for (pi, pj), val in c_coeffs.items():
            # W index is k*n + i. Here n=3.
            W3[pj * 3 + pi, idx] = val

    def verify_decomposition(U_arr, V_arr, W_arr):
        """Verify a decomposition is correct using multiple test matrices."""
        rank_used = U_arr.shape[1]
        np.random.seed(12345)
        max_error = 0
        for _ in range(5):
            A_t = np.random.randn(n, m)
            B_t = np.random.randn(m, p)
            C_exp = A_t @ B_t
            C_comp = np.zeros((n, p))
            for r in range(rank_used):
                M_u = np.dot(U_arr[:, r], A_t.flatten())
                M_v = np.dot(V_arr[:, r], B_t.flatten())
                M_r = M_u * M_v
                C_comp += M_r * W_arr[:, r].reshape(p, n).T
            max_error = max(max_error, np.max(np.abs(C_comp - C_exp)))
        return max_error

    def build_kronecker_decomposition(U3, V3, W3, U2, V2, W2):
        """
        Build 6x6 decomposition using Laderman (3x3) blocks and Strassen (2x2) sub-blocks.
        Rank = 23 * 7 = 161.
        """
        rank_3 = U3.shape[1]
        rank_2 = U2.shape[1]
        total_rank = rank_3 * rank_2
        
        U6 = np.zeros((36, total_rank), dtype=float)
        V6 = np.zeros((36, total_rank), dtype=float)
        W6 = np.zeros((36, total_rank), dtype=float)
        
        for r3 in range(rank_3):
            for r2 in range(rank_2):
                r = r3 * rank_2 + r2
                
                # U6 construction
                for I in range(3):
                    for i_local in range(2):
                        r_g = 2 * I + i_local
                        for J in range(3):
                            for j_local in range(2):
                                c_g = 2 * J + j_local
                                u_block_val = U3[3 * I + J, r3]
                                u_sub_val = U2[2 * i_local + j_local, r2]
                                U6[r_g * 6 + c_g, r] = u_block_val * u_sub_val
                                
                # V6 construction
                for J in range(3):
                    for j_local in range(2):
                        for K in range(3):
                            for k_local in range(2):
                                r_g = 2 * J + j_local
                                c_g = 2 * K + k_local
                                v_block_val = V3[3 * J + K, r3]
                                v_sub_val = V2[2 * j_local + k_local, r2]
                                V6[r_g * 6 + c_g, r] = v_block_val * v_sub_val
                                
                # W6 construction
                for K in range(3):
                    for k_local in range(2):
                        for I in range(3):
                            for i_local in range(2):
                                r_g = 2 * K + k_local
                                c_g = 2 * I + i_local
                                w_block_val = W3[3 * K + I, r3]
                                w_sub_val = W2[2 * k_local + i_local, r2]
                                W6[r_g * 6 + c_g, r] = w_block_val * w_sub_val
                                
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}, total_rank

    def build_strassen_6x6():
        """Fallback: Strassen 2x2 blocks of 3x3 schoolbook (Rank 189)."""
        U3_school = np.zeros((9, 27), dtype=int)
        V3_school = np.zeros((9, 27), dtype=int)
        W3_school = np.zeros((9, 27), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U3_school[i * 3 + j, r] = 1
                    V3_school[j * 3 + k, r] = 1
                    W3_school[k * 3 + i, r] = 1
                    
        return build_kronecker_decomposition(U3_school, V3_school, W3_school, U2, V2, W2)

    best_solution = None
    best_rank = float('inf')

    # Try Rank 161 (Laderman 3x3 + Strassen 2x2)
    if time.time() < deadline - 10:
        sol_161, rank_161 = build_kronecker_decomposition(U3, V3, W3, U2, V2, W2)
        U_t = np.array(sol_161["U"], dtype=float)
        V_t = np.array(sol_161["V"], dtype=float)
        W_t = np.array(sol_161["W"], dtype=float)
        err = verify_decomposition(U_t, V_t, W_t)
        if err < 1e-10:
            best_solution = sol_161
            best_rank = rank_161

    # Fallback to Rank 189 (Schoolbook 3x3 + Strassen 2x2)
    if best_solution is None or best_rank > 189:
        if time.time() < deadline - 5:
            sol_189, rank_189 = build_strassen_6x6()
            U_t = np.array(sol_189["U"], dtype=float)
            V_t = np.array(sol_189["V"], dtype=float)
            W_t = np.array(sol_189["W"], dtype=float)
            err = verify_decomposition(U_t, V_t, W_t)
            if err < 1e-10 and rank_189 < best_rank:
                best_solution = sol_189
                best_rank = rank_189

    # Final Fallback: Schoolbook 6x6 (Rank 216)
    if best_solution is None:
        rank_trivial = 216
        U_t = np.zeros((36, rank_trivial))
        V_t = np.zeros((36, rank_trivial))
        W_t = np.zeros((36, rank_trivial))
        idx = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U_t[i * 6 + j, idx] = 1
                    V_t[j * 6 + k, idx] = 1
                    W_t[k * 6 + i, idx] = 1
                    idx += 1
        best_solution = {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
        best_rank = rank_trivial

    return best_solution
# EVOLVE-BLOCK-END
