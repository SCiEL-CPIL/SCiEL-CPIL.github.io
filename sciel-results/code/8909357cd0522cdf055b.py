"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    deadline = time.time() + 140.0  # safety margin before 150s limit

    # WFTA decomposition: 2x4x6 via Z2 x Z3
    # Block A: Strassen 2x2x2 (rank 7) for i in {0,1}, j in {0,1}, k in {0,1}
    # Block B: WFTA 2x2x4 (rank 13) for i in {0,1}, j in {2,3}, k in {2,3,4,5}
    # Total rank: 7 + 13 = 20

    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # --- Block A: Strassen 2x2x2, indices i={0,1}, j={0,1}, k={0,1}, cols 0-6 ---
    def add_strassen(i0, j0, k0, off):
        U[i0*m+j0][off+0] += 1; U[(i0+1)*m+(j0+1)][off+0] += 1
        V[j0*p+k0][off+0] += 1; V[(j0+1)*p+(k0+1)][off+0] += 1
        W[k0*n+i0][off+0] += 1; W[(k0+1)*n+(i0+1)][off+0] += 1
        W[(k0+1)*n+i0][off+0] += 1; W[k0*n+(i0+1)][off+0] += 1

        U[(i0+1)*m+(j0+1)][off+1] += 1
        V[(j0+1)*p+(k0+1)][off+1] += 1; V[j0*p+(k0+1)][off+1] -= 1
        W[(k0+1)*n+(i0+1)][off+1] += 1

        U[i0*m+j0][off+2] += 1; U[i0*m+(j0+1)][off+2] += 1
        V[(j0+1)*p+(k0+1)][off+2] += 1
        W[(k0+1)*n+i0][off+2] += 1; W[(k0+1)*n+(i0+1)][off+2] += 1

        U[(i0+1)*m+j0][off+3] += 1; U[i0*m+j0][off+3] -= 1
        V[j0*p+k0][off+3] += 1; V[j0*p+(k0+1)][off+3] += 1
        W[k0*n+(i0+1)][off+3] += 1; W[(k0+1)*n+(i0+1)][off+3] += 1

        U[i0*m+j0][off+4] += 1; U[(i0+1)*m+(j0+1)][off+4] += 1
        V[(j0+1)*p+(k0+1)][off+4] += 1; V[(j0+1)*p+k0][off+4] -= 1
        W[(k0+1)*n+i0][off+4] += 1; W[(k0+1)*n+(i0+1)][off+4] += 1

        U[i0*m+(j0+1)][off+5] += 1; U[(i0+1)*m+(j0+1)][off+5] -= 1
        V[j0*p+k0][off+5] += 1; V[(j0+1)*p+k0][off+5] += 1
        W[k0*n+i0][off+5] += 1; W[k0*n+(i0+1)][off+5] += 1

        U[(i0+1)*m+j0][off+6] += 1; U[i0*m+j0][off+6] -= 1; U[(i0+1)*m+(j0+1)][off+6] -= 1
        V[(j0+1)*p+(k0+1)][off+6] += 1; V[(j0+1)*p+k0][off+6] += 1
        W[(k0+1)*n+i0][off+6] += 1; W[(k0+1)*n+(i0+1)][off+6] += 1

    add_strassen(0, 0, 0, 0)

    # --- Block B: WFTA on Z2 x Z3 for 2x2x4, indices i={0,1}, j={2,3}, k={2,3,4,5}, cols 7-19 ---
    # Local indices: i in {0,1}, j_local in {0,1} (j=2->0, j=3->1), k_local in {0,1,2,3} (k=2->0, k=3->1, k=4->2, k=5->3)
    # The 2x2x4 tensor can be decomposed as M_{2,2,2} ⊗ M_{1,1,2} but that gives rank 7*2=14.
    # Alternatively, use WFTA on Z2 x Z2 for the k part (splitting 4 into 2+2):
    # M_{2,2,4} = M_{2,2,2} ⊕ M_{2,2,2} (two 2x2x2 blocks), rank 7+7=14.
    # But we want rank 13.
    #
    # Actually, for the WFTA approach, let me use the known rank-13 construction for M_{2,2,3}
    # and handle the 4th k-index separately.
    #
    # M_{2,2,4} = M_{2,2,3} ⊕ M_{2,2,1}
    # M_{2,2,3} has rank 13 (WFTA on Z2 x Z3)
    # M_{2,2,1} has rank 4 (schoolbook: 2*2*1=4)
    # Total: 13 + 4 = 17. Not 13.
    #
    # I think the rank-20 construction uses a different split. Let me try:
    # M_{2,4,6} = M_{2,2,2} ⊕ M_{2,2,4} where M_{2,2,4} has rank 13.
    # But M_{2,2,4} = M_{2,2,2} ⊗ M_{1,1,2} has rank 14, not 13.
    #
    # I'm going to implement the WFTA for M_{2,2,3} (rank 13) and combine with
    # a separate handling of the remaining indices.
    #
    # For now, let me just use the rank-42 construction and return it.

    # Fallback: rank-42 Strassen construction
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1

        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1

        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1

        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)

    # Verify using numpy
    def verify_fast(U_mat, V_mat, W_mat):
        r = len(U_mat[0])
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(r):
                        actual += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    if not verify_fast(U, V, W):
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
