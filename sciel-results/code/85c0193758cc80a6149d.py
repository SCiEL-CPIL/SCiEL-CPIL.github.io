"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 3, 5, 6
    deadline = time.time() + 110.0

    # ---- Strassen 2x2x2 core: returns U,V,W of shape (4,7), (4,7), (4,7) ----
    # U: 4 inputs (a0,a1,a2,a3) for A = [[a0,a1],[a2,a3]]
    # V: 4 inputs (b0,b1,b2,b3) for B = [[b0,b1],[b2,b3]]
    # W: 4 outputs (c0,c1,c2,c3) for C = [[c0,c1],[c2,c3]]
    # The 7 Strassen intermediates:
    # M1 = (a0+a3)(b0+b3)
    # M2 = (a2+a3)(b0)
    # M3 = (a0)(b1+b3)
    # M4 = (a3)(b2)
    # M5 = (a0+a1)(b3)
    # M6 = (a2+a0)(b1+b0)
    # M7 = (a1+a3)(b2+b3)
    # C0 = M1 - M4 + M6 + M7
    # C1 = M3 + M5
    # C2 = M2 + M4
    # C3 = M1 - M3 + M5 + M6
    #
    # U[i, k]: coefficient of input a_i in intermediate k
    # V[j, k]: coefficient of input b_j in intermediate k
    # W[l, k]: coefficient of intermediate k in output c_l
    #
    # Inputs indexed: a0=(0,0), a1=(0,1), a2=(1,0), a3=(1,1)
    # b0=(0,0), b1=(0,1), b2=(1,0), b3=(1,1)
    # c0=(0,0), c1=(0,1), c2=(1,0), c3=(1,1)
    #
    # M1: a0+a3 times b0+b3 -> U: a0=1,a3=1; V: b0=1,b3=1
    # M2: a2+a3 times b0 -> U: a2=1,a3=1; V: b0=1
    # M3: a0 times b1+b3 -> U: a0=1; V: b1=1,b3=1
    # M4: a3 times b2 -> U: a3=1; V: b2=1
    # M5: a0+a1 times b3 -> U: a0=1,a1=1; V: b3=1
    # M6: a2+a0 times b1+b0 -> U: a0=1,a2=1; V: b0=1,b1=1
    # M7: a1+a3 times b2+b3 -> U: a1=1,a3=1; V: b2=1,b3=1
    #
    # C0 = M1 - M4 + M6 + M7
    # C1 = M3 + M5
    # C2 = M2 + M4
    # C3 = M1 - M3 + M5 + M6

    U_strassen = np.zeros((4, 7), dtype=int)
    V_strassen = np.zeros((4, 7), dtype=int)
    W_strassen = np.zeros((4, 7), dtype=int)
    # M1
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    # M2
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    # M3
    U_strassen[0, 2] = 1
    V_strassen[1, 2] = 1; V_strassen[3, 2] = 1
    # M4
    U_strassen[3, 3] = 1
    V_strassen[2, 3] = 1
    # M5
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[3, 4] = 1
    # M6
    U_strassen[0, 5] = 1; U_strassen[2, 5] = 1
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
    # M7
    U_strassen[1, 6] = 1; U_strassen[3, 6] = 1
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
    # W: C = M1 - M4 + M6 + M7; C1 = M3 + M5; C2 = M2 + M4; C3 = M1 - M3 + M5 + M6
    W_strassen[0, 0] = 1; W_strassen[0, 3] = -1; W_strassen[0, 5] = 1; W_strassen[0, 6] = 1
    W_strassen[1, 2] = 1; W_strassen[1, 4] = 1
    W_strassen[2, 1] = 1; W_strassen[2, 3] = 1
    W_strassen[3, 0] = 1; W_strassen[3, 2] = -1; W_strassen[3, 4] = 1; W_strassen[3, 5] = 1

    # ---- 3x3x3 Strassen-like decomposition: rank 14 is minimal known? Actually R(3,3,3)<=14 via Strassen embedding ----
    # We use the standard embedding: 3x3 = [[X, a],[b, c]] where X is 2x2, a is 2x1, b is 1x2, c is 1x1.
    # C = [[X,a],[b,c]] * [[Y, d],[e, f]]
    # C11 = XY + a*e, C12 = Xd + a*f, C21 = bY + c*e, C22 = bd + c*f
    # We can use Strassen for the 2x2 block products and handle cross terms.
    # 
    # Actually, a simpler approach: use the known result R(3,3,3) <= 14 (Strassen's original).
    # But implementing the full 14-term decomposition is complex. Let's use a simpler bound:
    # R(3,3,3) <= R(2,2,2) + 3*R(2,2,1) + 3*R(2,1,2) + 3*R(1,2,2) + R(1,1,1) ... this gets messy.
    # 
    # Alternative: use the trivial decomposition for the 3x3x3 part: rank 27.
    # Then total rank = 7 (for 2x2x2 part, repeated for each of 3 output cols... wait)
    #
    # Let me reconsider the approach. We have 3x5x6.
    # Split m=5 into 2+3.
    # Part 1: 3x2x6. Split p=6 into 3*2. For each 3x2x2 subproblem, use Strassen.
    #   3x2x2: A is 3x2, B is 2x2, C is 3x2.
    #   We can view this as: top 2 rows use Strassen (rank 7), bottom row: rank 2 (direct).
    #   So rank per 3x2x2 = 7 + 2 = 9.
    #   Total for 3x2x6 = 3 * 9 = 27.
    #
    # Part 2: 3x3x6. Split p=6 into 3*2. For each 3x3x2 subproblem:
    #   3x3x2: A is 3x3, B is 3x2, C is 3x2.
    #   Use Strassen on 2x2 blocks: 
    #     A = [[X, a],[b, c]] where X is 2x2, a is 2x1, b is 1x2, c is 1x1.
    #     B = [[Y, d],[e, f]] where Y is 2x2, d is 2x1, e is 1x2, f is 1x1.
    #     C11 = XY + a*e (2x2 product + rank-1 terms)
    #     C12 = Xd + a*f
    #     C21 = bY + c*e
    #     C22 = bd + c*f
    #   Using Strassen for XY: rank 7.
    #   a*e: rank 1 (outer product of 2-vectors... actually a is 2x1, e is 1x2, so a*e is 2x2 rank 1) -> 1 rank-1 term
    #   Xd: X is 2x2, d is 2x1 -> 2x1. Direct: rank 4. Or use Strassen-like: rank 3? 
    #   Actually for 2x2 times 2x1, the rank is at most 4 (direct). 
    #   This is getting complicated. Let me just use direct multiplication for the 3x3x2 part: rank 3*3*2 = 18.
    #   Total for 3x3x6 = 3 * 18 = 54.
    #   Grand total: 27 + 54 = 81.
    #
    # That's still better than 90. Let me implement this.

    # Actually, let me try a cleaner approach. Let me just use the direct method but with a smaller rank by exploiting structure.
    # The direct method gives 3*5*6 = 90.
    # 
    # A better idea: use the fact that for 3x5x6, we can potentially do better.
    # But let me just implement the block decomposition with Strassen for the 2x2 parts.
    #
    # Final plan:
    # Split m=5 into m1=2, m2=3.
    # Split p=6 into p1=2, p2=2, p3=2 (three chunks of 2).
    #
    # Part A: A[:, 0:2] * B[0:2, :] -> 3x2x6. For each p-chunk of size 2: 3x2x2.
    #   3x2x2: rows 0,1 use Strassen (rank 7), row 2 uses direct (rank 2). Total rank 9 per chunk.
    #   3 chunks: 27.
    #
    # Part B: A[:, 2:5] * B[2:5, :] -> 3x3x6. For each p-chunk of size 2: 3x3x2.
    #   3x3x2: direct rank 18 per chunk.
    #   3 chunks: 54.
    #
    # Total: 27 + 54 = 81.

    # Let me implement this.
    # U has shape (n*m, rank) = (15, rank)
    # V has shape (m*p, rank) = (30, rank)
    # W has shape (p*n, rank) = (18, rank)
    #
    # Indexing:
    # U[i*m + j, col] for i in 0..2, j in 0..4
    # V[j*p + k, col] for j in 0..4, k in 0..5
    # W[k*n + i, col] for k in 0..5, i in 0..2

    all_U = []
    all_V = []
    all_W = []

    # Part A: m-chunk 0:1 (j=0,1), for each p-chunk (k_offset = 0, 2, 4)
    for pc in range(3):
        k0 = pc * 2
        k1 = k0 + 1
        # Strassen for rows i=0,1 (2x2 block of A)
        # A_block = A[0:2, 0:2], B_block = B[0:2, k0:k0+2], C_block = C[0:2, k0:k0+2]
        # U_strassen maps (a0,a1,a2,a3) to 7 intermediates
        # a0=A[0,0], a1=A[0,1], a2=A[1,0], a3=A[1,1]
        # b0=B[0,k0], b1=B[0,k0+1], b2=B[1,k0], b3=B[1,k0+1]
        # c0=C[0,k0], c1=C[0,k0+1], c2=C[1,k0], c3=C[1,k0+1]
        for s in range(7):
            col = len(all_U)
            # U: only rows 0,1 of A (i=0,1) and j=0,1
            # U[i*m + j, col]
            # a0 -> i=0,j=0; a1 -> i=0,j=1; a2 -> i=1,j=0; a3 -> i=1,j=1
            u_val = U_strassen[:, s]
            for idx, val in enumerate(u_val):
                if val != 0:
                    a_i = idx // 2  # 0 or 1
                    a_j = idx % 2    # 0 or 1
                    all_U.append([0.0] * (col + 1))
                    all_U[col] = [0.0] * (col + 1)
            # This approach is getting messy. Let me use numpy arrays.
        # Let me restart with numpy.

    # Let me use numpy arrays for the whole thing.
    rank_total = 81
    U_mat = np.zeros((15, rank_total), dtype=float)
    V_mat = np.zeros((30, rank_total), dtype=float)
    W_mat = np.zeros((18, rank_total), dtype=float)

    col = 0

    # Part A: j=0,1 (first 2 of 5 middle dims), each p-chunk of 2
    for pc in range(3):
        k0 = pc * 2
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k0+2]
        for s in range(7):
            # U contributions
            for idx in range(4):
                val = U_strassen[idx, s]
                if val != 0:
                    a_i = idx // 2
                    a_j = idx % 2
                    U_mat[a_i * 5 + a_j, col] = val
            # V contributions
            for idx in range(4):
                val = V_strassen[idx, s]
                if val != 0:
                    b_j = idx // 2
                    b_k = idx % 2
                    V_mat[b_j * 6 + (k0 + b_k), col] = val
            # W contributions
            for idx in range(4):
                val = W_strassen[idx, s]
                if val != 0:
                    c_i = idx // 2
                    c_k = idx % 2
                    W_mat[(k0 + c_k) * 3 + c_i, col] = val
            col += 1

        # Direct for A[2, 0:2] * B[0:2, k0:k0+2]
        for j in range(2):
            for k_off in range(2):
                U_mat[2 * 5 + j, col] = 1
                V_mat[j * 6 + (k0 + k_off), col] = 1
                W_mat[(k0 + k_off) * 3 + 2, col] = 1
                col += 1

    # Part B: j=2,3,4 (last 3 of 5 middle dims), each p-chunk of 2, direct
    for pc in range(3):
        k0 = pc * 2
        for i in range(3):
            for j in range(3):
                j_abs = 2 + j
                for k_off in range(2):
                    U_mat[i * 5 + j_abs, col] = 1
                    V_mat[j_abs * 6 + (k0 + k_off), col] = 1
                    W_mat[(k0 + k_off) * 3 + i, col] = 1
                    col += 1

    assert col == rank_total, f"col={col}, expected {rank_total}"

    # Convert to lists of ints (all values are integers)
    U_list = [[int(x) for x in row] for row in U_mat]
    V_list = [[int(x) for x in row] for row in V_mat]
    W_list = [[int(x) for x in row] for row in W_mat]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
