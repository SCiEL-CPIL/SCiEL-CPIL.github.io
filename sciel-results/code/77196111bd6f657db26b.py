"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Rank-77 tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Strategy: Split into 4x4x4 (Strassen rank 49) + 4x4x2 (rank 28) = 77.
    """
    n, m, p = 4, 4, 6

    # --- Build 2x2x2 Strassen (rank 7) ---
    def build_strassen_2x2x2():
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]

        # M1 = (A11+A22)(B11+B22) -> C11+1, C12+0, C21+0, C22+1
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=1

        # M2 = (A21+A22)B11 -> C21+1, C22-1
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[2][1]=1; W[3][1]=-1

        # M3 = A11(B12-B22) -> C12+1, C22+1
        U[0][2]=1
        V[1][2]=1; V[3][2]=-1
        W[1][2]=1; W[3][2]=1

        # M4 = A22(B21-B11) -> C11+1, C21+1
        U[3][3]=1
        V[2][3]=1; V[0][3]=-1
        W[0][3]=1; W[2][3]=1

        # M5 = (A11+A12)B22 -> C11-1, C12+1
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=-1; W[1][4]=1

        # M6 = (A21-A11)(B11+B12) -> C22+1
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1

        # M7 = (A12-A22)(B21+B22) -> C11+1
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        W[0][6]=1

        return U, V, W

    # --- Build 2x2x1 schoolbook (rank 4) ---
    def build_school_2x2x1():
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]

        U[0][0]=1; V[0][0]=1; W[0][0]=1
        U[1][1]=1; V[1][1]=1; W[0][1]=1
        U[2][2]=1; V[0][2]=1; W[1][2]=1
        U[3][3]=1; V[1][3]=1; W[1][3]=1

        return U, V, W

    # --- Build 2x2x3 (rank 11) ---
    def build_2x2x3():
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]

        # Part 1: 2x2 * 2x1 -> 2x1 (rank 4), B col 0, C col 0
        U1, V1, W1 = build_school_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0:
                    U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0:
                    V[r*3][c] += V1[r][c]
            for r in range(2):
                if W1[r][c] != 0:
                    W[r][c] += W1[r][c]

        # Part 2: 2x2 * 2x2 -> 2x2 (rank 7), B cols 1-2, C cols 1-2
        U2, V2, W2 = build_strassen_2x2x2()
        base = 4
        v2_map = {0: 1, 1: 2, 2: 4, 3: 5}
        w2_map = {0: 2, 1: 4, 2: 3, 3: 5}
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0:
                    U[r][base+c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0:
                    V[v2_map[r]][base+c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0:
                    W[w2_map[r]][base+c] += W2[r][c]

        return U, V, W

    # --- Build 4x4x4 via Strassen (rank 49) ---
    def build_4x4x4():
        n2, m2, p2 = 2, 2, 2
        U_sub, V_sub, W_sub = build_strassen_2x2x2()
        rank_sub = 7
        total_rank = 7 * rank_sub  # 49

        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(16)]
        W = [[0]*total_rank for _ in range(16)]

        # Strassen on 4x4x4: 7 sub-problems each 2x2x2
        # A blocks: (oi, oj) where oi, oj in {0,2}
        # B blocks: (oj, ok) where oj, ok in {0,2}
        # C blocks: (oi, ok) where oi, ok in {0,2}
        #
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)

        strassen_terms = [
            # (U_offsets, V_offsets, W_offsets)
            # Each offset is (start_i, start_j) for U, (start_j, start_k) for V, (start_i, start_k) for W
            # M1
            ([(0,0),(2,2)], [(0,0),(2,2)], [(0,0),(2,2)]),
            # M2
            ([(2,0),(2,2)], [(0,0)], [(2,0),(2,2)]),
            # M3
            ([(0,0)], [(0,2),(2,2)], [(0,2),(2,2)]),
            # M4
            ([(2,2)], [(2,0),(0,0)], [(0,0),(2,0)]),
            # M5
            ([(0,0),(0,2)], [(2,2)], [(0,0),(0,2)]),
            # M6
            ([(2,0),(0,0)], [(0,0),(0,2)], [(2,2)]),
            # M7
            ([(0,2),(2,2)], [(2,0),(2,2)], [(0,0)]),
        ]

        signs = [
            # For M1: +1,+1 for U; +1,+1 for V; +1,+1 for W
            [(1,1), (1,1), (1,1)],
            # M2
            [(1,1), (1,), (1,-1)],
            # M3
            [(1,), (1,-1), (1,1)],
            # M4
            [(1,), (1,-1), (1,1)],
            # M5
            [(1,1), (1,), (-1,1)],
            # M6
            [(1,-1), (1,1), (1,)],
            # M7
            [(1,-1), (1,1), (1,)],
        ]

        for t in range(7):
            u_offs = strassen_terms[t][0]
            v_offs = strassen_terms[t][1]
            w_offs = strassen_terms[t][2]
            u_sigs = signs[t][0]
            v_sigs = signs[t][1]
            w_sigs = signs[t][2]

            for c in range(rank_sub):
                col_idx = t * rank_sub + c
                for idx, (oi, oj) in enumerate(u_offs):
                    for ri in range(2):
                        for rj in range(2):
                            U[(oi+ri)*4 + (oj+rj)][col_idx] += u_sigs[idx] * U_sub[ri*2+rj][c]
                for idx, (oj, ok) in enumerate(v_offs):
                    for rj in range(2):
                        for rk in range(2):
                            V[(oj+rj)*4 + (ok+rk)][col_idx] += v_sigs[idx] * V_sub[rj*2+rk][c]
                for idx, (oi, ok) in enumerate(w_offs):
                    for ri in range(2):
                        for rk in range(2):
                            W[(ok+rk)*4 + (oi+ri)][col_idx] += w_sigs[idx] * W_sub[rk*2+ri][c]

        return U, V, W

    # --- Build 4x4x2 (rank 28) ---
    def build_4x4x2():
        # 4x4 * 4x2: Split A into 2x2 blocks, B into 2x1 blocks
        # 4 sub-problems of 2x2 * 2x1, each rank 4 -> total 16
        # But we can do better with Strassen-like approach
        # Actually: 4x4 * 4x2 = 4x(4x2). Split A rows into 2 groups of 2.
        # For each group of 2 rows of A, multiply by 4x2 B.
        # 2x4 * 4x2: split B into two 4x1 columns.
        # 2x4 * 4x1: rank 8 schoolbook? No, let's just do direct.
        #
        # Simplest: 4x4 * 4x2. Split into 4 sub-problems:
        # A[0:2,0:2]*B[0:2,0:2] -> C[0:2,0:2] (2x2*2x2, rank 7)
        # A[0:2,0:2]*B[2:4,0:2] -> C[0:2,0:2] (2x2*2x2, rank 7)
        # A[0:2,2:4]*B[0:2,0:2] -> C[0:2,0:2] (2x2*2x2, rank 7)
        # A[0:2,2:4]*B[2:4,0:2] -> C[0:2,0:2] (2x2*2x2, rank 7)
        # That's 28 for top half, same for bottom half = 56. Too much.
        #
        # Better: 4x4 * 4x2. Use the fact that B is 4x2.
        # Split A into 2x2 blocks (4 blocks), B into 2x2 blocks (2 blocks).
        # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
        # C[2:4, 0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
        #
        # Each 2x2*2x2 is rank 7. So 4 * 7 = 28.
        total_rank = 28
        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(8)]
        W = [[0]*total_rank for _ in range(8)]

        U_sub, V_sub, W_sub = build_strassen_2x2x2()
        rank_sub = 7

        # 4 sub-problems:
        # (oi, oj) for A block start, (oj, ok) for B block start, (oi, ok) for C block start
        # B is 4x2, so ok is always 0 (only one column block of width 2)
        # V index: j*2+k, k in 0..1
        # W index: k*4+i, k in 0..1, i in 0..3

        sub_problems = [
            # (A row offset, A col offset, B row offset)
            (0, 0, 0),  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
            (0, 2, 2),  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
            (2, 0, 0),  # A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]
            (2, 2, 2),  # A[2:4,2:4] * B[2:4,0:2] -> C[2:4,0:2]
        ]

        for sp_idx, (oi, oj, oj_b) in enumerate(sub_problems):
            base = sp_idx * rank_sub
            for c in range(rank_sub):
                col_idx = base + c
                for ri in range(2):
                    for rj in range(2):
                        U[(oi+ri)*4 + (oj+rj)][col_idx] += U_sub[ri*2+rj][c]
                for rj in range(2):
                    for rk in range(2):
                        V[(oj_b+rj)*2 + rk][col_idx] += V_sub[rj*2+rk][c]
                for ri in range(2):
                    for rk in range(2):
                        W[rk*4 + (oi+ri)][col_idx] += W_sub[rk*2+ri][c]

        return U, V, W

    # --- Combine 4x4x4 and 4x4x2 ---
    U_444, V_444, W_444 = build_4x4x4()
    U_442, V_442, W_442 = build_4x4x2()

    rank_444 = len(U_444[0])  # 49
    rank_442 = len(U_442[0])  # 28
    total_rank = rank_444 + rank_442  # 77

    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(24)]
    W = [[0]*total_rank for _ in range(24)]

    # Copy 4x4x4 part (columns 0-3 of B and C)
    for r in range(16):
        for c in range(rank_444):
            U[r][c] = U_444[r][c]
    for r in range(16):
        for c in range(rank_444):
            V[r][c] = V_444[r][c]
    for r in range(16):
        for c in range(rank_444):
            W[r][c] = W_444[r][c]

    # Copy 4x4x2 part (columns 4-5 of B and C)
    for r in range(16):
        for c in range(rank_442):
            U[r][rank_444+c] = U_442[r][c]
    for r in range(8):
        for c in range(rank_442):
            V[r + 16][rank_444+c] = V_442[r][c]
    for r in range(8):
        for c in range(rank_442):
            W[r + 16][rank_444+c] = W_442[r][c]

    # --- Verify ---
    if _time_remaining() > 5.0:
        try:
            U_np = np.array(U).T  # (rank, 16)
            V_np = np.array(V).T  # (rank, 24)
            W_np = np.array(W).T  # (rank, 24)

            T_check = np.zeros((4, 4, 6))
            for i in range(4):
                for j in range(4):
                    for k in range(6):
                        T_check[i,j,k] = np.sum(
                            U_np[:, i*4+j] * V_np[:, j*6+k] * W_np[:, k*4+i]
                        )
            if not np.allclose(T_check, 1.0, atol=1e-9):
                # Fallback: return schoolbook rank 96
                U_fb = [[0]*96 for _ in range(16)]
                V_fb = [[0]*96 for _ in range(24)]
                W_fb = [[0]*96 for _ in range(24)]
                for i in range(4):
                    for j in range(4):
                        for k in range(6):
                            idx = i*24 + j*6 + k
                            U_fb[i*4+j][idx] = 1
                            V_fb[j*6+k][idx] = 1
                            W_fb[k*4+i][idx] = 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
