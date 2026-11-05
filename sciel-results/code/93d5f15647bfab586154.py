import numpy as np
import time

def construct():
    t0 = time.time()
    tlimit = t0 + 115

    # Dimensions
    nn, mm, pp = 4, 5, 6
    NU = nn * mm          # 20
    NV = mm * pp          # 30
    NW = pp * nn          # 24

    # Strassen 2x2x2: 7 rank-1 terms
    # Each term: (U_2x2, V_2x2, W_2x2)
    STR = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,-1]]),
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,0],[1,1]]),
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,1],[0,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[1,0]]),
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]

    # Build list of (u_idx, v_idx, w_idx, coeff) for each rank-1 term
    # Strategy: partition 4x5x6 as:
    #   Part A: 4x4x6 -> two 2x4x6 blocks -> four 2x2x6 -> eight 2x2x3
    #     Actually let's do: 4x4x6 = sum over (di,dk) in {0,1}x{0,1,2} of 2x4x2 blocks
    #     Then each 2x4x2 = sum over dj in {0,1} of 2x2x2
    #     So 4x4x6 = 8 blocks of 2x2x2, each using 7 Strassen terms = 56 terms
    #   Part B: 4x1x6 -> schoolbook = 24 terms
    #   Total rank = 56 + 24 = 80

    # Actually let me be more careful. The tensor is T[i,j,k] where i in 0..3, j in 0..4, k in 0..5.
    # We want to decompose T = sum_r u_r ⊗ v_r ⊗ w_r.

    # For the 4x4x6 part (i in 0..3, j in 0..3, k in 0..5):
    #   Split i into two groups of 2: i0 = i//2 (0 or 1), i1 = i%2
    #   Split j into two groups of 2: j0 = j//2 (0 or 1), j1 = j%2
    #   Split k into three groups of 2: k0 = k//2 (0,1,2), k1 = k%2
    #   For each (i0, j0, k0) in {0,1}x{0,1}x{0,1,2}: use Strassen on the 2x2x2 block
    #   That's 2*2*3 = 12 blocks, each with 7 terms = 84 terms

    # For the 4x1x6 part (i in 0..3, j=4, k in 0..5): schoolbook = 4*1*6 = 24 terms
    # Total rank = 84 + 24 = 108

    # Let me verify this matches the recursive approach's count.
    # The recursive approach: compute_rank(4,5,6):
    #   ni=4 >= mj=5? No. mj=5 >= ni=4 and mj>=pk=6? No (5<6). pk=6 >= ni=4 and pk>=mj=5? Yes.
    #   So: compute_rank(4,5,2) + compute_rank(4,5,4)
    #   compute_rank(4,5,2): ni=4>=mj=5? No. mj=5>=ni=4 and mj>=pk=2? Yes.
    #     compute_rank(4,2,2) + compute_rank(4,3,2)
    #     compute_rank(4,2,2): ni=4>=mj=2 and ni>=pk=2 and ni>2:
    #       compute_rank(2,2,2) + compute_rank(2,2,2) = 7+7 = 14
    #     compute_rank(4,3,2): ni=4>=mj=3 and ni>=pk=2 and ni>2:
    #       compute_rank(2,3,2) + compute_rank(2,3,2)
    #       compute_rank(2,3,2): ni=2,mj=3,pk=2. ni<=2,mj<=2? No (mj=3). 
    #         ni>=mj? No. mj>=ni and mj>=pk and mj>2? Yes (3>=2,3>=2,3>2):
    #           compute_rank(2,2,2) + compute_rank(2,1,2) = 7 + 2*1*2 = 7+4 = 11
    #       So compute_rank(4,3,2) = 11+11 = 22
    #     compute_rank(4,5,2) = 14+22 = 36
    #   compute_rank(4,5,4): ni=4>=mj=5? No. mj=5>=ni=4 and mj>=pk=4 and mj>2? Yes.
    #     compute_rank(4,2,4) + compute_rank(4,3,4)
    #     compute_rank(4,2,4): ni=4>=mj=2 and ni>=pk=4? 4>=4 yes, ni>2:
    #       compute_rank(2,2,4) + compute_rank(2,2,4)
    #       compute_rank(2,2,4): ni=2,mj=2,pk=4. ni<=2,mj<=2 but pk>2. 
    #         ni>=mj? 2>=2 yes, ni>=pk? 2>=4 no. mj>=ni? 2>=2 yes, mj>=pk? 2>=4 no. pk>=ni? 4>=2 yes, pk>=mj? 4>=2 yes, pk>2:
    #           compute_rank(2,2,2) + compute_rank(2,2,2) = 7+7 = 14
    #       So compute_rank(4,2,4) = 14+14 = 28
    #     compute_rank(4,3,4): ni=4>=mj=3 and ni>=pk=4? 4>=4 yes, ni>2:
    #       compute_rank(2,3,4) + compute_rank(2,3,4)
    #       compute_rank(2,3,4): ni=2,mj=3,pk=4. 
    #         ni>=mj? No. mj>=ni and mj>=pk? 3>=2 yes, 3>=4 no. pk>=ni and pk>=mj and pk>2? 4>=2,4>=3,4>2: yes
    #           compute_rank(2,3,2) + compute_rank(2,3,2) = 11+11 = 22
    #       So compute_rank(4,3,4) = 22+22 = 44
    #     compute_rank(4,5,4) = 28+44 = 72
    #   Total = 36+72 = 108. Good.

    # So rank is 108. Let me build it differently: use a flat list of contributions.
    # Each contribution is (i,j,k,coeff) meaning T[i,j,k] += coeff * (u_term * v_term * w_term)
    # But actually each rank-1 term contributes to ALL (i,j,k) via u[i,j]*v[j,k]*w[k,i].
    # So I need to track which (i,j,k) each rank-1 term covers and with what coefficient.

    # Better approach: build U, V, W matrices directly by iterating over the Strassen blocks.

    RANK = 108
    U = np.zeros((NU, RANK), dtype=np.float64)
    V = np.zeros((NV, RANK), dtype=np.float64)
    W = np.zeros((NW, RANK), dtype=np.float64)

    col = 0

    # Part 1: 4x4x6 region, decomposed into 12 Strassen blocks
    # Block (bi, bj, bk): i in [2*bi, 2*bi+1], j in [2*bj, 2*bj+1], k in [2*bk, 2*bk+1]
    for bi in range(2):
        for bj in range(2):
            for bk in range(3):
                for (u_mat, v_mat, w_mat) in STR:
                    # u_mat: 2x2, maps (local_i, local_j) -> coefficient for U
                    for li in range(2):
                        for lj in range(2):
                            c = u_mat[li][lj]
                            if c != 0:
                                gi = 2 * bi + li
                                gj = 2 * bj + lj
                                U[gi * mm + gj, col] += c
                    # v_mat: 2x2, maps (local_j, local_k) -> coefficient for V
                    for lj in range(2):
                        for lk in range(2):
                            c = v_mat[lj][lk]
                            if c != 0:
                                gj = 2 * bj + lj
                                gk = 2 * bk + lk
                                V[gj * pp + gk, col] += c
                    # w_mat: 2x2, maps (local_k, local_i) -> coefficient for W
                    for lk in range(2):
                        for li in range(2):
                            c = w_mat[lk][li]
                            if c != 0:
                                gk = 2 * bk + lk
                                gi = 2 * bi + li
                                W[gk * nn + gi, col] += c
                    col += 1

    # Part 2: 4x1x6 region (j=4), schoolbook
    for gi in range(4):
        for gk in range(6):
            U[gi * mm + 4, col] = 1.0
            V[4 * pp + gk, col] = 1.0
            W[gk * nn + gi, col] = 1.0
            col += 1

    assert col == RANK, f"col={col}, expected {RANK}"

    # Verify: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # Use vectorized verification
    Ua = U[:, :col]
    Va = V[:, :col]
    Wa = W[:, :col]

    max_err = 0.0
    for i in range(nn):
        for j in range(mm):
            for k in range(pp):
                ui = Ua[i * mm + j, :]
                vj = Va[j * pp + k, :]
                wk = Wa[k * nn + i, :]
                s = np.dot(ui, vj * wk)
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-9:
        # Fallback: pure schoolbook
        rs = nn * mm * pp
        Us = np.zeros((NU, rs), dtype=np.float64)
        Vs = np.zeros((NV, rs), dtype=np.float64)
        Ws = np.zeros((NW, rs), dtype=np.float64)
        c2 = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    Us[i * mm + j, c2] = 1.0
                    Vs[j * pp + k, c2] = 1.0
                    Ws[k * nn + i, c2] = 1.0
                    c2 += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    return {"U": Ua.tolist(), "V": Va.tolist(), "W": Wa.tolist()}
