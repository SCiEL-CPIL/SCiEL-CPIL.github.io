"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Strategy: Decompose 2x5x6 into 2x1x6 (rank 12) + 2x4x6 (rank 42)
    # 2x1x6: each term is independent, rank = 2*1*6 = 12
    # 2x4x6: use 3 Strassen 2x2x2 blocks on j-pairs {0,1},{2,3} and k-pairs {0,1},{2,3},{4,5}
    #   2x2x6 can be decomposed with rank 21 (3 Strassen instances, saving 3 from 24)
    #   So 2x4x6 = 2 * (2x2x6) = 2 * 21 = 42
    # Total rank = 12 + 42 = 54

    rank = 54

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: 2x1x6 block (j=0, all i, k)
    # For i in {0,1}, j=0, k in {0,...,5}: standard terms
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    # Part 1 done: col should be 12
    if col != 12:
        raise ValueError(f"Part 1: expected col=12, got {col}")

    # Part 2: 2x4x6 block (j in {1,2,3,4}, i in {0,1}, k in {0,...,5})
    # Split j into two pairs: j_pair1 = {1,2}, j_pair2 = {3,4}
    # For each j_pair, split k into three pairs: k_pair1 = {0,1}, k_pair2 = {2,3}, k_pair3 = {4,5}
    # Each (i_pair={0,1}, j_pair, k_pair) is a 2x2x2 Strassen block with rank 7

    def add_strassen_block(i_base, j_base, k_base, col_offset):
        """Add a 2x2x2 Strassen block starting at column col_offset.
        i_base: 0 or 1 (base for i index, actual i = i_base + 0 or 1)
        j_base: base for j index, actual j = j_base + 0 or 1
        k_base: base for k index, actual k = k_base + 0 or 1
        Returns the new column offset (col_offset + 7).
        """
        # Local indices
        u00 = i_base * m + j_base
        u01 = i_base * m + (j_base + 1)
        u10 = (i_base + 1) * m + j_base
        u11 = (i_base + 1) * m + (j_base + 1)
        
        v00 = j_base * p + k_base
        v01 = j_base * p + (k_base + 1)
        v10 = (j_base + 1) * p + k_base
        v11 = (j_base + 1) * p + (k_base + 1)
        
        w00 = k_base * n + i_base
        w01 = (k_base + 1) * n + i_base
        w10 = k_base * n + (i_base + 1)
        w11 = (k_base + 1) * n + (i_base + 1)
        
        c = col_offset
        
        # M1 = (A00 + A11)(B00 + B11) -> C00 + C11
        for u_idx, coeff in [(u00, 1), (u11, 1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v00, 1), (v11, 1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w00, 1), (w11, 1)]:
            W[w_idx][c] += coeff
        c += 1
        
        # M2 = (A10 + A11)B00 -> C10 - C11
        for u_idx, coeff in [(u10, 1), (u11, 1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v00, 1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w10, 1), (w11, -1)]:
            W[w_idx][c] += coeff
        c += 1
        
        # M3 = A00(B01 - B11) -> C01 + C11
        for u_idx, coeff in [(u00, 1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v01, 1), (v11, -1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w01, 1), (w11, 1)]:
            W[w_idx][c] += coeff
        c += 1
        
        # M4 = A11(B10 - B00) -> C00 + C10
        for u_idx, coeff in [(u11, 1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v10, 1), (v00, -1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w00, 1), (w10, 1)]:
            W[w_idx][c] += coeff
        c += 1
        
        # M5 = (A00 + A01)B11 -> -C00 + C01
        for u_idx, coeff in [(u00, 1), (u01, 1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v11, 1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w00, -1), (w01, 1)]:
            W[w_idx][c] += coeff
        c += 1
        
        # M6 = (A10 - A00)(B00 + B01) -> C11
        for u_idx, coeff in [(u10, 1), (u00, -1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v00, 1), (v01, 1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w11, 1)]:
            W[w_idx][c] += coeff
        c += 1
        
        # M7 = (A01 - A11)(B10 + B11) -> C00
        for u_idx, coeff in [(u01, 1), (u11, -1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v10, 1), (v11, 1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w00, 1)]:
            W[w_idx][c] += coeff
        c += 1
        
        return c

    # Apply Strassen blocks for j in {1,2,3,4}
    j_pairs = [(1, 2), (3, 4)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            # i_base is always 0 for the 2x2x2 block since n=2
            col = add_strassen_block(0, j1, k1, col)

    if col != 54:
        raise ValueError(f"Expected col=54, got {col}")

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback: return a rank-60 trivial decomposition
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
