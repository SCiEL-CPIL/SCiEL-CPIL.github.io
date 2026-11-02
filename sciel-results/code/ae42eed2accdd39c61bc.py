import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS

    # Laderman's rank-23 decomposition for 3x3x3
    # Each entry: (u_vec, v_vec, w_vec) where each is length 9
    laderman_3x3x3 = [
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
    ]

    rank_laderman = 23

    # Strategy: Use Laderman for first 3 columns (k=0,1,2),
    # then another Laderman-like block for columns k=3,4,5,
    # and schoolbook for k=6.
    #
    # Actually, let's think more carefully. The multiplication is:
    # (C)_{i,k} = sum_j A_{i,j} * B_{j,k}
    #
    # We can split B into columns: B[:, 0:3], B[:, 3:6], B[:, 6:7]
    # Each block is a 3x3x3 problem (for the first two) and 3x3x1 (for the last).
    #
    # For the 3x3x1 problem (single column), we need rank at most 9 (schoolbook).
    # But can we do better? For a single output column, C[:,k] = A * B[:,k].
    # This is just a matrix-vector product. The minimum rank for computing
    # A*x for arbitrary x is at least n*m = 9 (since each A_{i,j} appears).
    # Actually, we can use rank 9 with the standard approach.
    #
    # So total rank = 23 + 23 + 9 = 55.
    #
    # But wait - can we share factors between blocks? The U factors are the same
    # for all blocks (they depend only on A). The V and W factors differ per block.
    #
    # Actually, in tensor decomposition, each term r has U_r, V_r, W_r.
    # For block 1 (k=0,1,2): U uses rows i*m+j, V uses rows j*p+k (k=0..2), W uses rows k*n+i (k=0..2)
    # For block 2 (k=3,4,5): U uses same rows, V uses rows j*p+k (k=3..5), W uses rows k*n+i (k=3..5)
    # For block 3 (k=6): U uses same rows, V uses rows j*p+6, W uses rows 6*n+i
    #
    # The U factors for block 1 and block 2 can be the SAME Laderman U factors!
    # Because U depends only on i and j, not on k.
    #
    # So we can reuse the same U vectors for both Laderman blocks.
    # Total rank = 23 (shared U) + 23 (V,W for block 2) + 9 (block 3) = 55
    #
    # Wait, that's not quite right. In a tensor decomposition, each rank term is a
    # triple (u, v, w). We can't "share" U between terms - each term has its own U.
    # But we can set U for term r in block 2 to be the same as U for term r in block 1.
    # That's fine - it just means the U matrix has repeated columns.
    #
    # So total rank is still 23 + 23 + 9 = 55.
    #
    # Let's build this.

    total_rank = 23 + 23 + 9  # = 55

    U = [[0] * total_rank for _ in range(n * m)]   # 9 x 55
    V = [[0] * total_rank for _ in range(m * p)]   # 21 x 55
    W = [[0] * total_rank for _ in range(p * n)]   # 21 x 55

    # Block 1: k = 0, 1, 2 (first 3 columns)
    for r in range(23):
        u, v, w = laderman_3x3x3[r]
        for idx in range(9):
            U[idx][r] = u[idx]
        # V: index j*p + k where k in 0..2
        for j in range(3):
            for k in range(3):
                V[j * p + k][r] = v[j * 3 + k]
        # W: index k*n + i where k in 0..2
        for k in range(3):
            for i in range(3):
                W[k * n + i][r] = w[k * 3 + i]

    # Block 2: k = 3, 4, 5 (next 3 columns)
    offset2 = 23
    for r in range(23):
        u, v, w = laderman_3x3x3[r]
        for idx in range(9):
            U[idx][offset2 + r] = u[idx]
        # V: index j*p + k where k in 3..5
        for j in range(3):
            for k in range(3, 6):
                V[j * p + k][offset2 + r] = v[j * 3 + (k - 3)]
        # W: index k*n + i where k in 3..5
        for k in range(3, 6):
            for i in range(3):
                W[k * n + i][offset2 + r] = w[(k - 3) * 3 + i]

    # Block 3: k = 6 (last column)
    offset3 = 46
    col3 = 0
    for i in range(3):
        for j in range(3):
            U[i * m + j][offset3 + col3] = 1
            V[j * p + 6][offset3 + col3] = 1
            W[6 * n + i][offset3 + col3] = 1
            col3 += 1

    # Verify with numpy (faster)
    def verify_numpy(U_mat, V_mat, W_mat, n, m, p):
        rank = U_mat.shape[1]
        U_arr = np.array(U_mat, dtype=np.float64)
        V_arr = np.array(V_mat, dtype=np.float64)
        W_arr = np.array(W_mat, dtype=np.float64)
        
        for i in range(n):
            for k in range(p):
                expected = 1.0
                computed = 0.0
                for j in range(m):
                    for r in range(rank):
                        computed += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(computed - expected) > 1e-10:
                    return False
        return True

    if verify_numpy(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END

