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
    # Each pattern: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict (i,j) -> coeff for A[i,j]
    # B_coeffs: dict (j,k) -> coeff for B[j,k]
    # C_coeffs: dict (i,k) -> coeff for C[i,k]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    laderman_23_patterns = [
        # 1: a00*(b00+b10+b20) -> c00
        ({(0,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1}),
        # 2: a10*(b10+b20) -> c10
        ({(1,0):1}, {(1,0):1,(2,0):1}, {(1,0):1}),
        # 3: a20*(b20) -> c20
        ({(2,0):1}, {(2,0):1}, {(2,0):1}),
        # 4: (a11+a21)*(b01+b11+b21) -> c10, c20
        ({(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,0):1,(2,0):-1}),
        # 5: a11*(b21) -> c10
        ({(1,1):1}, {(2,1):1}, {(1,0):-1}),
        # 6: a21*(b21) -> c20
        ({(2,1):1}, {(2,1):1}, {(2,0):1}),
        # 7: (a12+a22)*(b02+b12+b22) -> c10, c20
        ({(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,0):-1,(2,0):1}),
        # 8: a12*(b22) -> c10
        ({(1,2):1}, {(2,2):1}, {(1,0):1}),
        # 9: a22*(b22) -> c20
        ({(2,2):1}, {(2,2):1}, {(2,0):-1}),
        # 10: a01*(b00+b10+b20) -> c01
        ({(0,1):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,1):1}),
        # 11: a02*(b00+b10+b20) -> c02
        ({(0,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,2):1}),
        # 12: a11*(b01+b11+b21) -> c11
        ({(1,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1}),
        # 13: a12*(b02+b12+b22) -> c12
        ({(1,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1}),
        # 14: a21*(b01+b11+b21) -> c21
        ({(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(2,1):1}),
        # 15: a22*(b02+b12+b22) -> c22
        ({(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(2,2):1}),
        # 16: a00*(b11+b21) -> c01
        ({(0,0):1}, {(1,1):1,(2,1):1}, {(0,1):1}),
        # 17: a00*(b12+b22) -> c02
        ({(0,0):1}, {(1,2):1,(2,2):1}, {(0,2):1}),
        # 18: a11*(b22) -> c12
        ({(1,1):1}, {(2,2):1}, {(1,2):-1}),
        # 19: a12*(b21) -> c11
        ({(1,2):1}, {(2,1):1}, {(1,1):-1}),
        # 20: a21*(b22) -> c22
        ({(2,1):1}, {(2,2):1}, {(2,2):-1}),
        # 21: a22*(b21) -> c21
        ({(2,2):1}, {(2,1):1}, {(2,1):-1}),
        # 22: (a11+a22)*(b01+b12) -> c11, c22
        ({(1,1):1,(2,2):1}, {(0,1):1,(1,2):1}, {(1,1):1,(2,2):1}),
        # 23: (a12+a21)*(b02+b11) -> c12, c21
        ({(1,2):1,(2,1):1}, {(0,2):1,(1,1):1}, {(1,2):1,(2,1):1}),
    ]

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

    def build_winograd_strassen_6x6():
        """Build 6x6x6 decomposition: Winograd 3x3 (rank 8) x Strassen 2x2 (rank 7) = rank 192."""
        # Winograd's 3x3 algorithm: rank 8
        # C = A*B where A, B, C are 3x3
        # Uses 8 multiplications
        winograd_8_patterns = [
            # m1 = (a00 + a11 + a22) * (b00 + b11 + b22)
            ({(0,0):1, (1,1):1, (2,2):1},
             {(0,0):1, (1,1):1, (2,2):1},
             {(0,0):1, (1,1):1, (2,2):1}),
            # m2 = (a01 + a12) * (b01 + b12)
            ({(0,1):1, (1,2):1},
             {(0,1):1, (1,2):1},
             {(0,1):1, (1,2):1}),
            # m3 = (a10 + a21) * (b10 + b21)
            ({(1,0):1, (2,1):1},
             {(1,0):1, (2,1):1},
             {(1,0):1, (2,1):1}),
            # m4 = (a00 - a11) * (b11 - b00)
            ({(0,0):1, (1,1):-1},
             {(1,1):1, (0,0):-1},
             {(0,1):-1, (1,0):1}),
            # m5 = (a11 - a22) * (b22 - b11)
            ({(1,1):1, (2,2):-1},
             {(2,2):1, (1,1):-1},
             {(1,2):-1, (2,1):1}),
            # m6 = (a22 - a00) * (b00 - b22)
            ({(2,2):1, (0,0):-1},
             {(0,0):1, (2,2):-1},
             {(2,0):-1, (0,2):1}),
            # m7 = (a01 - a21 + a22) * (b22 + b12 - b21)
            ({(0,1):1, (2,1):-1, (2,2):1},
             {(2,2):1, (1,2):1, (2,1):-1},
             {(2,0):1, (0,2):-1}),
            # m8 = (a00 + a10 - a11) * (b11 - b01 + b00)
            ({(0,0):1, (1,0):1, (1,1):-1},
             {(1,1):1, (0,1):-1, (0,0):1},
             {(0,0):1, (1,0):-1}),
        ]
        # Verify Winograd patterns produce correct 3x3 product
        def verify_winograd():
            np.random.seed(123)
            A_t = np.random.randn(3, 3)
            B_t = np.random.randn(3, 3)
            C_expected = A_t @ B_t
            C_computed = np.zeros((3, 3))
            for u_c, v_c, w_c in winograd_8_patterns:
                mu = sum(v * A_t[r, c] for (r, c), v in u_c.items())
                mv = sum(v * B_t[r, c] for (r, c), v in v_c.items())
                for (r, c), v in w_c.items():
                    C_computed[r, c] += v * mu * mv
            return np.max(np.abs(C_computed - C_expected))
        
        err_w = verify_winograd()
        if err_w > 1e-10:
            return None, 0  # Winograd patterns invalid
        
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
        
        # Kronecker: 3x3 blocks for rows/cols of A (dim 6), 3x3 for B rows, 3x3 for C cols
        # 2x2 blocks within each 3x3 sub-block
        # Actually: split 6 into 3 groups of 2. Use Winograd on the 3x3 "super" structure,
        # and Strassen within each 2x2 sub-block.
        # 
        # The 6x6 matrix is viewed as a 3x3 block matrix where each block is 2x2.
        # Winograd gives 8 multiplications of 3x3 block-matrices.
        # Each block-matrix multiplication (2x2 blocks) uses Strassen (7 scalar mults).
        # Total rank = 8 * 7 = 56? No wait...
        #
        # Actually the Kronecker product approach:
        # The tensor for 6x6 = tensor for 3x3 ⊗ tensor for 2x2
        # rank(6x6) <= rank(3x3) * rank(2x2) = 8 * 7 = 56
        # But this isn't right either. Let me think more carefully.
        #
        # The matrix multiplication tensor M_{n} has rank r_n.
        # M_{ab} = M_a ⊗ M_b, and r(M_a ⊗ M_b) <= r(M_a) * r(M_b).
        # So r(M_6) = r(M_{3*2}) <= r(M_3) * r(M_2) = 8 * 7 = 56.
        #
        # Wait, that's the rank of the tensor, not the number of scalar multiplications
        # in a block algorithm. Let me reconsider.
        #
        # The Kronecker product of decompositions:
        # If M_3 = sum_{i=1}^{8} u_i ⊗ v_i ⊗ w_i
        # and M_2 = sum_{j=1}^{7} u'_j ⊗ v'_j ⊗ w'_j
        # Then M_6 = M_3 ⊗ M_2 = sum_{i,j} (u_i⊗u'_j) ⊗ (v_i⊗v'_j) ⊗ (w_i⊗w'_j)
        # This gives rank 8*7 = 56.
        #
        # But the indices: M_3 acts on 3x3 matrices (9 entries each), M_2 on 2x2 (4 entries).
        # M_6 acts on 6x6 matrices (36 entries each).
        # The Kronecker: index (a,b) where a in 1..9 (flattened 3x3), b in 1..4 (flattened 2x2)
        # gives 36 total entries. So the flattening is: 6x6 matrix viewed as 3x3 blocks of 2x2.
        # Entry (I*2+pi, J*2+pj) in the 6x6 corresponds to block (I,J) position (pi,pj).
        
        rank = 8 * 7  # 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col_idx = 0
        for wi, (u_c, v_c, w_c) in enumerate(winograd_8_patterns):
            for sj, (su_c, sv_c, sw_c) in enumerate(strassen_patterns):
                # u_c: dict (I, J) -> val for the 3x3 super-structure
                # su_c: dict (pi, pj) -> val for the 2x2 sub-block
                # Combined: U[row, col] where row = (I*2+pi)*m + (J*2+pj)
                for (I, J), val_u in u_c.items():
                    for (pi, pj), val_su in su_c.items():
                        r = I * 2 + pi
                        c = J * 2 + pj
                        U[r * m + c, col_idx] += val_u * val_su
                
                for (I, J), val_v in v_c.items():
                    for (pi, pj), val_sv in sv_c.items():
                        r = I * 2 + pi
                        c = J * 2 + pj
                        V[r * p + c, col_idx] += val_v * val_sv
                
                for (I, J), val_w in w_c.items():
                    for (pi, pj), val_sw in sw_c.items():
                        i = I * 2 + pi
                        k = J * 2 + pj
                        W[k * n + i, col_idx] += val_w * val_sw
                
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

        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]

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

    # Test matrices for verification
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    # Try all known constructions and pick the lowest-rank valid one
    best_solution = None
    best_rank = float('inf')

    # 1) Winograd 3x3 x Strassen 2x2 Kronecker (rank 56)
    if time.time() < deadline - 20:
        ws_sol, ws_rank = build_winograd_strassen_6x6()
        if ws_sol is not None and ws_rank > 0:
            U_ws = np.array(ws_sol["U"], dtype=float)
            V_ws = np.array(ws_sol["V"], dtype=float)
            W_ws = np.array(ws_sol["W"], dtype=float)
            err_ws = verify_decomposition(U_ws, V_ws, W_ws, A_test, B_test, C_expected)
            if err_ws < 1e-10:
                best_solution = ws_sol
                best_rank = ws_rank

    # 2) Laderman-based 6x6 (rank 184)
    if best_solution is None or time.time() < deadline - 20:
        laderman_sol, laderman_rank = build_laderman_6x6()
        U_l = np.array(laderman_sol["U"], dtype=float)
        V_l = np.array(laderman_sol["V"], dtype=float)
        W_l = np.array(laderman_sol["W"], dtype=float)
        err_l = verify_decomposition(U_l, V_l, W_l, A_test, B_test, C_expected)
        if err_l < 1e-10 and laderman_rank < best_rank:
            best_solution = laderman_sol
            best_rank = laderman_rank

    # 3) Strassen-based 6x6 (rank 189)
    if best_solution is None or time.time() < deadline - 20:
        strassen_sol, strassen_rank = build_strassen_6x6()
        U_s = np.array(strassen_sol["U"], dtype=float)
        V_s = np.array(strassen_sol["V"], dtype=float)
        W_s = np.array(strassen_sol["W"], dtype=float)
        err_s = verify_decomposition(U_s, V_s, W_s, A_test, B_test, C_expected)
        if err_s < 1e-10 and strassen_rank < best_rank:
            best_solution = strassen_sol
            best_rank = strassen_rank

    # If both failed (shouldn't happen), use trivial schoolbook
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
