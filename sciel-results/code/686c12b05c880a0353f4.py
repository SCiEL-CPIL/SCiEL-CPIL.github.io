"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    def verify_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    def verify_decomposition_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_np[i * m + j, :]
                    v_val = V_np[j * p + k, :]
                    w_val = W_np[k * n + i, :]
                    val = np.dot(u_val, v_val * w_val)
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Build the Strassen-based rank-44 decomposition
    rank = 44
    cols = []

    def add_term(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        cols.append((u_vec, v_vec, w_vec))

    def add_strassen(a, b, c, d, e, f, g, h, w11, w12, w21, w22):
        add_term([(a, 1), (d, 1)], [(e, 1), (h, 1)], [(w11, 1), (w22, 1)])
        add_term([(c, 1), (d, 1)], [(e, 1)], [(w21, 1), (w22, -1)])
        add_term([(a, 1)], [(f, 1), (h, -1)], [(w12, 1), (w22, 1)])
        add_term([(d, 1)], [(g, 1), (e, -1)], [(w11, 1), (w21, 1)])
        add_term([(a, 1), (b, 1)], [(h, 1)], [(w12, 1), (w11, -1)])
        add_term([(c, 1), (a, -1)], [(e, 1), (f, 1)], [(w22, 1)])
        add_term([(b, 1), (d, -1)], [(g, 1), (h, 1)], [(w11, 1)])

    def add_rank4(a, b, c, d, e, f, w1, w2):
        add_term([(a, 1)], [(c, 1)], [(w1, 1)])
        add_term([(b, 1)], [(e, 1)], [(w1, 1)])
        add_term([(a, 1)], [(d, 1)], [(w2, 1)])
        add_term([(b, 1)], [(f, 1)], [(w2, 1)])

    # A (3x4): Row0=[0,1,2,3], Row1=[4,5,6,7], Row2=[8,9,10,11]
    # B (4x4): Row0=[0,1,2,3], Row1=[4,5,6,7], Row2=[8,9,10,11], Row3=[12,13,14,15]
    # C (3x4) -> W[k*n+i]: C[0,0]=0,C[0,1]=3,C[0,2]=6,C[0,3]=9
    #                     C[1,0]=1,C[1,1]=4,C[1,2]=7,C[1,3]=10
    #                     C[2,0]=2,C[2,1]=5,C[2,2]=8,C[2,3]=11

    add_strassen(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)
    add_strassen(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)
    add_rank4(8, 9, 0, 1, 4, 5, 2, 5)
    add_rank4(8, 9, 2, 3, 6, 7, 8, 11)
    add_strassen(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)
    add_strassen(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)
    add_rank4(10, 11, 8, 9, 12, 13, 2, 5)
    add_rank4(10, 11, 10, 11, 14, 15, 8, 11)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        u_vec, v_vec, w_vec = cols[r]
        for i in range(n * m):
            U[i][r] = u_vec[i]
        for j in range(m * p):
            V[j][r] = v_vec[j]
        for k in range(p * n):
            W[k][r] = w_vec[k]

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    if verify_decomposition_fast(U_np, V_np, W_np, rank):
        best_U, best_V, best_W = U_np, V_np, W_np
        best_rank = rank

    # Now try to reduce rank by attempting to merge/eliminate terms
    # Strategy: For each pair of columns, check if their sum/difference can be absorbed
    # into other columns, effectively reducing rank by 1.
    # We use the fact that if col_a + col_b = 0 in tensor space, we can eliminate one.
    # More generally, try to find linear dependencies among the 44 rank-1 tensors.

    if time.time() > deadline - 20:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try to find if any single column is a linear combination of others
    # This is expensive, so we do a limited search
    # Represent each column as its triple (u_vec, v_vec, w_vec)
    # The tensor contribution of column c is: T_c = u_c ⊗ v_c ⊗ w_c (outer product)
    # We need: sum_c T_c = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} (the matmul tensor)

    # Alternative approach: try replacing pairs of schoolbook terms in the rank-4 parts
    # with fewer terms using Strassen-like tricks on smaller blocks.
    # For 1x2x2, the minimum rank is 4 (schoolbook), so no improvement there.
    # For 2x2x2, Strassen gives 7 < 8.

    # Try a different partition: use 2x4x4 blocks
    # A is 3x4. Split as: top 2 rows (2x4) and bottom 1 row (1x4).
    # B is 4x4. Split B into two 4x2 halves: left (4x2) and right (4x2).
    # C is 3x4.

    # Block (2x4) * (4x2) -> (2x2): This is 2x4x2.
    # We can split the 2x4 A-block into two 2x2 blocks, each multiplied by the 4x2 B-block.
    # Each 2x2 * 4x2 can be done as: split B into two 2x2 vertical blocks.
    # So 2x2 * 2x2 (Strassen, rank 7) + 2x2 * 2x2 (Strassen, rank 7) = rank 14 for each 2x4*4x2.
    # Two such blocks: 28.
    # Then 1x4 * 4x2 -> 1x2: rank 4 (schoolbook) for each half, total 8.
    # Grand total: 28 + 8 = 36? Let me verify.

    # Actually: A_top (2x4) * B_left (4x2) -> C_top_left (2x2)
    #   Split A_top into A_top_L (2x2) and A_top_R (2x2)
    #   Split B_left into B_top (2x2) and B_bot (2x2)
    #   C_top_left[0:2,0] = A_top_L * B_top[:,0] + A_top_R * B_bot[:,0]... no this is getting complicated.

    # Let me try a cleaner approach: use the known result that R(3,4,4) might be 40 or lower.
    # Try a different Strassen embedding.

    # Alternative: Treat the problem as (2x4 + 1x4) * (4x4) -> (2x4 + 1x4)
    # Top 2 rows: 2x4 * 4x4 -> 2x4, done as two 2x4*4x2 blocks.
    # Each 2x4*4x2: split into 2*(2x2*2x2) using Strassen = 2*7 = 14
    # So top part: 2*14 = 28
    # Bottom row: 1x4 * 4x4 -> 1x4, done as 4*(1x1*1x1) = 4, or 2*(1x2*2x2) = 2*4 = 8
    # Total: 28 + 8 = 36

    # Let me build this decomposition.
    if time.time() > deadline - 15:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    cols2 = []

    def add_term2(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        cols2.append((u_vec, v_vec, w_vec))

    def add_strassen2(a, b, c, d, e, f, g, h, w11, w12, w21, w22):
        add_term2([(a, 1), (d, 1)], [(e, 1), (h, 1)], [(w11, 1), (w22, 1)])
        add_term2([(c, 1), (d, 1)], [(e, 1)], [(w21, 1), (w22, -1)])
        add_term2([(a, 1)], [(f, 1), (h, -1)], [(w12, 1), (w22, 1)])
        add_term2([(d, 1)], [(g, 1), (e, -1)], [(w11, 1), (w21, 1)])
        add_term2([(a, 1), (b, 1)], [(h, 1)], [(w12, 1), (w11, -1)])
        add_term2([(c, 1), (a, -1)], [(e, 1), (f, 1)], [(w22, 1)])
        add_term2([(b, 1), (d, -1)], [(g, 1), (h, 1)], [(w11, 1)])

    def add_rank4_1x2x2(a, b, c, d, e, f, w1, w2):
        add_term2([(a, 1)], [(c, 1)], [(w1, 1)])
        add_term2([(b, 1)], [(e, 1)], [(w1, 1)])
        add_term2([(a, 1)], [(d, 1)], [(w2, 1)])
        add_term2([(b, 1)], [(f, 1)], [(w2, 1)])

    # Top 2 rows of A: indices 0,1,2,3 (row0) and 4,5,6,7 (row1)
    # B: 4x4, rows 0-3, cols 0-3
    # C top: rows 0,1 of C, cols 0-3

    # Split B into left half (cols 0,1) and right half (cols 2,3)
    # B_left: B[0:4, 0:2], B_right: B[0:4, 2:4]

    # For A_top (2x4) * B_left (4x2) -> C_top (2x2, cols 0,1)
    # Split A_top into A_top_L (2x2, cols 0,1) and A_top_R (2x2, cols 2,3)
    # Split B_left into B_top (2x2, rows 0,1) and B_bot (2x2, rows 2,3)
    # C_top = A_top_L * B_top + A_top_R * B_bot (but B is 4x2, so B_top is 2x2, B_bot is 2x2)

    # A_top_L: A[0:2, 0:2] = indices 0,1,4,5
    # A_top_R: A[0:2, 2:4] = indices 2,3,6,7
    # B_top: B[0:2, 0:2] = indices 0,1,4,5
    # B_bot: B[2:4, 0:2] = indices 8,9,12,13
    # C_top: C[0:2, 0:2] -> W: C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4

    add_strassen2(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)  # A_top_L * B_top
    add_strassen2(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)  # A_top_R * B_bot

    # For A_top (2x4) * B_right (4x2) -> C_top (2x2, cols 2,3)
    # B_right_top: B[0:2, 2:4] = indices 2,3,6,7
    # B_right_bot: B[2:4, 2:4] = indices 10,11,14,15
    # C_top_right: C[0:2, 2:4] -> W: C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10

    add_strassen2(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)  # A_top_L * B_right_top
    add_strassen2(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)  # A_top_R * B_right_bot

    # Bottom row of A: indices 8,9,10,11
    # A_bot (1x4) * B_left (4x2) -> C_bot (1x2, cols 0,1)
    # This is 1x4*4x2 = 1x2, schoolbook rank 4
    # C_bot_left: C[2,0]=2, C[2,1]=5

    add_rank4_1x2x2(8, 9, 0, 1, 4, 5, 2, 5)  # A_bot_left * B_left

    # A_bot (1x4) * B_right (4x2) -> C_bot (1x2, cols 2,3)
    # C_bot_right: C[2,2]=8, C[2,3]=11

    add_rank4_1x2x2(10, 11, 2, 3, 6, 7, 8, 11)  # A_bot_right * B_right

    rank2 = len(cols2)
    if rank2 < best_rank:
        U2 = [[0] * rank2 for _ in range(n * m)]
        V2 = [[0] * rank2 for _ in range(m * p)]
        W2 = [[0] * rank2 for _ in range(p * n)]
        for r in range(rank2):
            u_vec, v_vec, w_vec = cols2[r]
            for i in range(n * m):
                U2[i][r] = u_vec[i]
            for j in range(m * p):
                V2[j][r] = v_vec[j]
            for k in range(p * n):
                W2[k][r] = w_vec[k]

        U2_np = np.array(U2, dtype=float)
        V2_np = np.array(V2, dtype=float)
        W2_np = np.array(W2, dtype=float)

        if verify_decomposition_fast(U2_np, V2_np, W2_np, rank2):
            best_U, best_V, best_W = U2_np, V2_np, W2_np
            best_rank = rank2

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
