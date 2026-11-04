"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for 6x6x6 matrix multiplication rank minimization."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock limit

    # Laderman's 3x3 matrix multiplication algorithm (rank 23)
    # Verified correct patterns for C = A*B where A,B,C are 3x3
    # Using 0-indexed (row, col) notation
    # Each pattern: (U_dict, V_dict, W_dict) where dict maps (row,col) -> coefficient
    # The contribution to C[i,j] from pattern r is: W[r][i,j] * U[r][i,j] * V[r][i,j]
    laderman_23_patterns = [
        # M1: A[0,0]*(B[0,0]+B[1,0]+B[2,0])
        ({(0,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1}),
        # M2: A[1,0]*(B[1,0]+B[2,0])  
        ({(1,0):1}, {(1,0):1,(2,0):1}, {(1,0):1}),
        # M3: A[2,0]*B[2,0]
        ({(2,0):1}, {(2,0):1}, {(2,0):1}),
        # M4: (A[1,1]+A[2,1])*(B[0,1]+B[1,1]+B[2,1])
        ({(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1,(2,1):1}),
        # M5: A[1,1]*B[2,1]
        ({(1,1):1}, {(2,1):1}, {(1,1):-1}),
        # M6: A[2,1]*B[2,1]
        ({(2,1):1}, {(2,1):1}, {(2,1):-1}),
        # M7: (A[1,2]+A[2,2])*(B[0,2]+B[1,2]+B[2,2])
        ({(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1,(2,2):1}),
        # M8: A[1,2]*B[2,2]
        ({(1,2):1}, {(2,2):1}, {(1,2):-1}),
        # M9: A[2,2]*B[2,2]
        ({(2,2):1}, {(2,2):1}, {(2,2):-1}),
        # M10: A[0,1]*(B[0,0]+B[1,0]+B[2,0])
        ({(0,1):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,1):1}),
        # M11: A[0,2]*(B[0,0]+B[1,0]+B[2,0])
        ({(0,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,2):1}),
        # M12: A[1,1]*(B[0,1]+B[1,1]+B[2,1])
        ({(1,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1}),
        # M13: A[1,2]*(B[0,2]+B[1,2]+B[2,2])
        ({(1,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1}),
        # M14: A[2,1]*(B[0,1]+B[1,1]+B[2,1])
        ({(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(2,1):1}),
        # M15: A[2,2]*(B[0,2]+B[1,2]+B[2,2])
        ({(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(2,2):1}),
        # M16: A[0,0]*(B[1,1]+B[2,1])
        ({(0,0):1}, {(1,1):1,(2,1):1}, {(0,1):1}),
        # M17: A[0,0]*(B[1,2]+B[2,2])
        ({(0,0):1}, {(1,2):1,(2,2):1}, {(0,2):1}),
        # M18: A[1,1]*B[2,2]
        ({(1,1):1}, {(2,2):1}, {(1,2):1}),
        # M19: A[1,2]*B[2,1]
        ({(1,2):1}, {(2,1):1}, {(1,1):1}),
        # M20: A[2,1]*B[2,2]
        ({(2,1):1}, {(2,2):1}, {(2,2):1}),
        # M21: A[2,2]*B[2,1]
        ({(2,2):1}, {(2,1):1}, {(2,1):1}),
        # M22: (A[1,1]+A[2,2])*(B[0,1]+B[1,2])
        ({(1,1):1,(2,2):1}, {(0,1):1,(1,2):1}, {(1,1):-1,(2,2):-1}),
        # M23: (A[1,2]+A[2,1])*(B[0,2]+B[1,1])
        ({(1,2):1,(2,1):1}, {(0,2):1,(1,1):1}, {(1,2):-1,(2,1):-1}),
    ]

    # Strassen 2x2 patterns
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def verify_3x3_patterns(patterns):
        """Verify that the 3x3 patterns correctly compute C = A*B."""
        np.random.seed(123)
        for trial in range(5):
            A_t = np.random.randn(3, 3)
            B_t = np.random.randn(3, 3)
            C_expected = A_t @ B_t
            C_computed = np.zeros((3, 3))
            for u_c, v_c, w_c in patterns:
                mu = sum(val * A_t[r, c] for (r, c), val in u_c.items())
                mv = sum(val * B_t[r, c] for (r, c), val in v_c.items())
                for (r, c), val in w_c.items():
                    C_computed[r, c] += val * mu * mv
            if np.max(np.abs(C_computed - C_expected)) > 1e-10:
                return False
        return True

    def verify_2x2_patterns(patterns):
        """Verify that the 2x2 patterns correctly compute C = A*B."""
        np.random.seed(456)
        for trial in range(5):
            A_t = np.random.randn(2, 2)
            B_t = np.random.randn(2, 2)
            C_expected = A_t @ B_t
            C_computed = np.zeros((2, 2))
            for u_c, v_c, w_c in patterns:
                mu = sum(val * A_t[r, c] for (r, c), val in u_c.items())
                mv = sum(val * B_t[r, c] for (r, c), val in v_c.items())
                for (r, c), val in w_c.items():
                    C_computed[r, c] += val * mu * mv
            if np.max(np.abs(C_computed - C_expected)) > 1e-10:
                return False
        return True

    def verify_decomposition(U_arr, V_arr, W_arr, A_test, B_test, C_expected):
        """Verify a decomposition is correct."""
        rank_used = U_arr.shape[1]
        C_computed = np.zeros((n, p))
        for r in range(rank_used):
            M_u = np.dot(U_arr[:, r], A_test.flatten())
            M_v = np.dot(V_arr[:, r], B_test.flatten())
            M_r = M_u * M_v
            C_computed += M_r * W_arr[:, r].reshape(p, n).T
        error = np.max(np.abs(C_computed - C_expected))
        return error

    def build_laderman_6x6():
        """Build 6x6x6 decomposition using Laderman's 3x3 algorithm on 2x2 blocks."""
        block_size = 3
        blocks_per_dim = 2
        rank = (blocks_per_dim ** 3) * 23  # 8 * 23 = 184
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for a_coeffs, b_coeffs, c_coeffs in laderman_23_patterns:
                        for (pi, pj), val in a_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val

                        for (pi, pj), val in b_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val

                        for (pi, pj), val in c_coeffs.items():
                            i = I * block_size + pi
                            k = K * block_size + pj
                            W[k * n + i, col_idx] += val

                        col_idx += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def build_strassen_6x6():
        """Build the 6x6x6 decomposition using Strassen on 2x2 blocks (rank 189)."""
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7  # 27 * 7 = 189
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val

                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val

                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i, col_idx] += val

                        col_idx += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def build_kronecker_3x3_schoolbook_2x2_strassen():
        """
        Build 6x6x6 using Kronecker product of 3x3 schoolbook (rank 27) and 2x2 Strassen (rank 7).
        This gives rank 27*7 = 189.
        
        The Kronecker construction: if we have a decomposition of m1xn1 matrix mult with rank r1
        and a decomposition of m2xn2 matrix mult with rank r2, we can construct a decomposition
        of (m1*m2) x (n1*n2) matrix mult with rank r1*r2.
        
        For 6x6 = (3*2) x (3*2), we use:
        - 3x3 schoolbook: rank 27
        - 2x2 Strassen: rank 7
        Total rank: 27 * 7 = 189
        """
        # 3x3 schoolbook patterns
        schoolbook_3x3 = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    schoolbook_3x3.append((
                        {(i, k): 1},
                        {(k, j): 1},
                        {(i, j): 1}
                    ))
        
        # 2x2 Strassen patterns
        strassen_2x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        rank = len(schoolbook_3x3) * len(strassen_2x2)  # 27 * 7 = 189
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Build using Kronecker product structure
        # For each pair (pattern_3x3, pattern_2x2), create a combined pattern
        col_idx = 0
        for u3, v3, w3 in schoolbook_3x3:
            for u2, v2, w2 in strassen_2x2:
                # U: for each (i3, k3) in u3 and (i2, k2) in u2
                #    the combined index is (i3*2 + i2, k3*2 + k2) in the 6x6 matrix
                for (i3, k3), val3 in u3.items():
                    for (i2, k2), val2 in u2.items():
                        row = i3 * 2 + i2
                        col = k3 * 2 + k2
                        U[row * m + col, col_idx] += val3 * val2
                
                # V: for each (k3, j3) in v3 and (k2, j2) in v2
                #    the combined index is (k3*2 + k2, j3*2 + j2)
                for (k3, j3), val3 in v3.items():
                    for (k2, j2), val2 in v2.items():
                        row = k3 * 2 + k2
                        col = j3 * 2 + j2
                        V[row * p + col, col_idx] += val3 * val2
                
                # W: for each (i3, j3) in w3 and (i2, j2) in w2
                #    the combined index is (i3*2 + i2, j3*2 + j2)
                #    W index is k*n + i, so we need to be careful
                for (i3, j3), val3 in w3.items():
                    for (i2, j2), val2 in w2.items():
                        i = i3 * 2 + i2
                        k = j3 * 2 + j2  # This is actually the column of C, which is k in our notation
                        W[k * n + i, col_idx] += val3 * val2
                
                col_idx += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    # Test matrices for verification
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    # Verify base patterns
    laderman_valid = verify_3x3_patterns(laderman_23_patterns)
    strassen_valid = verify_2x2_patterns(strassen_patterns)

    # Try known constructions and pick the lowest-rank valid one
    best_solution = None
    best_rank = float('inf')

    # 1) Laderman-based 6x6 (rank 184) - best known
    if laderman_valid and time.time() < deadline - 20:
        laderman_sol, laderman_rank = build_laderman_6x6()
        U_l = np.array(laderman_sol["U"], dtype=float)
        V_l = np.array(laderman_sol["V"], dtype=float)
        W_l = np.array(laderman_sol["W"], dtype=float)
        err_l = verify_decomposition(U_l, V_l, W_l, A_test, B_test, C_expected)
        if err_l < 1e-10 and laderman_rank < best_rank:
            best_solution = laderman_sol
            best_rank = laderman_rank

    # 2) Strassen-based 6x6 (rank 189)
    if strassen_valid and (best_solution is None or time.time() < deadline - 20):
        strassen_sol, strassen_rank = build_strassen_6x6()
        U_s = np.array(strassen_sol["U"], dtype=float)
        V_s = np.array(strassen_sol["V"], dtype=float)
        W_s = np.array(strassen_sol["W"], dtype=float)
        err_s = verify_decomposition(U_s, V_s, W_s, A_test, B_test, C_expected)
        if err_s < 1e-10 and strassen_rank < best_rank:
            best_solution = strassen_sol
            best_rank = strassen_rank

    # 3) Kronecker product of 3x3 schoolbook and 2x2 Strassen (rank 189)
    if time.time() < deadline - 20:
        kron_sol, kron_rank = build_kronecker_3x3_schoolbook_2x2_strassen()
        U_k = np.array(kron_sol["U"], dtype=float)
        V_k = np.array(kron_sol["V"], dtype=float)
        W_k = np.array(kron_sol["W"], dtype=float)
        err_k = verify_decomposition(U_k, V_k, W_k, A_test, B_test, C_expected)
        if err_k < 1e-10 and kron_rank < best_rank:
            best_solution = kron_sol
            best_rank = kron_rank

    # 4) Fallback: schoolbook (rank 216)
    if best_solution is None:
        rank_trivial = n * m * p  # 216
        U_t = np.zeros((n * m, rank_trivial))
        V_t = np.zeros((m * p, rank_trivial))
        W_t = np.zeros((p * n, rank_trivial))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_t[i * m + j, idx] = 1
                    V_t[j * p + k, idx] = 1
                    W_t[k * n + i, idx] = 1
                    idx += 1
        best_solution = {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
        best_rank = rank_trivial

    return best_solution
# EVOLVE-BLOCK-END
