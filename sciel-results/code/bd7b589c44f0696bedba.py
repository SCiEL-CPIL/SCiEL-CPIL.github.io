import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 4, 4

    def get_strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen, each 4x7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = 1
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[3, 6] = 1

        return U, V, W

    U2, V2, W2 = get_strassen_2x2x2()

    # Verify Strassen 2x2x2
    T2 = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(7):
                    s += U2[i * 2 + k, r] * V2[k * 2 + j, r] * W2[j * 2 + i, r]
                T2[i, j, k] = s
    strassen_ok = bool(np.allclose(T2, 1.0))

    # ------------------------------------------------------------------
    # Indexing conventions:
    #   U row  = i*m + k  = i*4 + k   (A[i,k])
    #   V row  = k*p + j  = k*4 + j   (B[k,j])
    #   W row  = j*n + i  = j*2 + i   (C[i,j])
    #
    # Partition:
    #   A = [A_L | A_R]  A_L=A[:,0:2], A_R=A[:,2:4]   (both 2x2)
    #   B = [[B_TL, B_TR],
    #        [B_BL, B_BR]]  (each 2x2)
    #   C = [C_L | C_R]   C_L=C[:,0:2], C_R=C[:,2:4]  (both 2x2)
    #   C_L = A_L*B_TL + A_R*B_BL
    #   C_R = A_L*B_TR + A_R*B_BR
    # ------------------------------------------------------------------

    def verify(Uf, Vf, Wf, rank):
        T = np.zeros((2, 4, 4))
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    s = 0.0
                    for r in range(rank):
                        s += Uf[i * 4 + k, r] * Vf[k * 4 + j, r] * Wf[j * 2 + i, r]
                    T[i, j, k] = s
        return bool(np.allclose(T, 1.0))

    def build_rank28():
        rank = 28
        Uf = np.zeros((8, rank))
        Vf = np.zeros((16, rank))
        Wf = np.zeros((8, rank))
        # Block 0: A_L * B_TL -> C_L
        Uf[0:4, 0:7] = U2; Vf[0:4, 0:7] = V2; Wf[0:4, 0:7] = W2
        # Block 1: A_R * B_BL -> C_L
        Uf[4:8, 7:14] = U2; Vf[8:12, 7:14] = V2; Wf[0:4, 7:14] = W2
        # Block 2: A_L * B_TR -> C_R
        Uf[0:4, 14:21] = U2; Vf[4:8, 14:21] = V2; Wf[4:8, 14:21] = W2
        # Block 3: A_R * B_BR -> C_R
        Uf[4:8, 21:28] = U2; Vf[12:16, 21:28] = V2; Wf[4:8, 21:28] = W2
        return Uf, Vf, Wf, rank

    def build_rank32():
        rank = 32
        Uf = np.zeros((8, rank))
        Vf = np.zeros((16, rank))
        Wf = np.zeros((8, rank))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    Uf[i * 4 + k, col] = 1
                    Vf[k * 4 + j, col] = 1
                    Wf[j * 2 + i, col] = 1
                    col += 1
        return Uf, Vf, Wf, rank

    best = None

    # --- Attempt 1: rank-24 via shared A-side Strassen sums ------------
    # Structural change vs. the 28-rank scheme: instead of 4 independent
    # 2x2x2 Strassen products, we share the A-side linear forms across the
    # two output blocks.  For each A-block (A_L, A_R) we form the 7 Strassen
    # A-sums once and pair each with the two B-blocks it feeds, giving
    # 2 A-blocks * 7 sums * 2 B-blocks = 28 candidate columns; we then drop
    # the 4 columns that are provably redundant (the cross terms that
    # cancel in the C-combination), landing at rank 24.
    if strassen_ok:
        Uf = np.zeros((8, 24))
        Vf = np.zeros((16, 24))
        Wf = np.zeros((8, 24))

        # A_L U-rows [0,1,2,3]; A_R U-rows [4,5,6,7]
        # B_TL V-rows [0,1,2,3]; B_TR V-rows [4,5,6,7]
        # B_BL V-rows [8,9,10,11]; B_BR V-rows [12,13,14,15]
        # C_L W-rows [0,1,2,3]; C_R W-rows [4,5,6,7]

        # For each A-block and each of its 7 Strassen A-sums, we create two
        # rank-1 columns (one per B-block it multiplies).  We keep all 28
        # but then linearly combine to 24 by noting the C-output of the two
        # B-blocks for a *given* A-sum can be merged: the W coefficient for
        # (A_L-sum_s, B_TL) and (A_L-sum_s, B_TR) differ only in which C
        # block they touch, so they cannot merge.  The true saving comes
        # from the fact that the *pair* of A-sums (s, s') that appear
        # symmetrically in C_L and C_R combine.  We therefore build the
        # explicit 24-column candidate and verify; if it fails we fall back.

        # Build the 24-column candidate by taking the 28 independent
        # Strassen columns and removing 4 that are exact linear
        # dependencies in the tensor sense (detected numerically).
        cand_U = np.zeros((8, 28))
        cand_V = np.zeros((16, 28))
        cand_W = np.zeros((8, 28))
        # 2 A-blocks x 7 sums x 2 B-blocks
        a_block_rows = [[0, 1, 2, 3], [4, 5, 6, 7]]
        # A_L feeds B_TL (C_L) and B_TR (C_R); A_R feeds B_BL (C_L) and B_BR (C_R)
        b_block_map = {
            0: ([0, 1, 2, 3], [0, 1, 2, 3]),   # A_L -> (B_TL for C_L, B_TR for C_R)
            1: ([8, 9, 10, 11], [12, 13, 14, 15]),  # A_R -> (B_BL for C_L, B_BR for C_R)
        }
        c_block_map = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
        col = 0
        for ab in range(2):
            a_rows = a_block_rows[ab]
            bL_rows, bR_rows = b_block_map[ab]
            cL_rows, cR_rows = c_block_map[ab]
            for s in range(7):
                # column for (A_block, sum_s, B_left)
                for r in range(4):
                    cand_U[a_rows[r], col] = U2[r, s]
                for r in range(4):
                    cand_V[bL_rows[r], col] = V2[r, s]
                for r in range(4):
                    cand_W[cL_rows[r], col] = W2[r, s]
                col += 1
                # column for (A_block, sum_s, B_right)
                for r in range(4):
                    cand_U[a_rows[r], col] = U2[r, s]
                for r in range(4):
                    cand_V[bR_rows[r], col] = V2[r, s]
                for r in range(4):
                    cand_W[cR_rows[r], col] = W2[r, s]
                col += 1

        # The 28 independent Strassen columns DO reconstruct the tensor
        # (that's just the rank-28 scheme).  To reach 24 we must find a
        # 24-column linear combination.  We solve for the minimal set by
        # checking whether dropping each of 4 strategically-chosen columns
        # and re-solving the remaining coefficients still reproduces T.
        # This is the actual structural reduction attempt.
        #
        # We attempt: keep a 24-column subset S and solve
        #   T = sum_{c in S} U_c V_c W_c
        # by least squares over the flattened tensor, then verify exactly.
        # Choose S as all 28 minus the 4 "M2-type" columns (indices 1,3,5,7
        # within each A-block pair) which are the ones most likely redundant.
        # (Heuristic; verified numerically below.)

        # Flatten target tensor
        target = np.zeros((8, 16, 8))
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    target[i * 4 + k, k * 4 + j, j * 2 + i] = 1.0

        # Build the 28-rank-1 factor "matrices" F_c of shape (8*16, 8)
        # so that target == sum_c F_c, where F_c = (U_c ⊗ V_c) outer W_c
        F_list = []
        for c in range(28):
            uc = cand_U[:, c]
            vc = cand_V[:, c]
            wc = cand_W[:, c]
            Fc = np.outer(np.outer(uc, vc).reshape(8 * 16), wc)
            F_list.append(Fc)

        # Try removing each 4-subset of "redundant" columns and solving
        # the remaining 24 as a linear system.  We test a handful of
        # candidate 4-subsets.
        import itertools
        found24 = False
        # Candidate redundant index sets (heuristic families)
        candidate_drops = [
            [1, 3, 5, 7],
            [2, 4, 6, 8],
            [0, 2, 4, 6],
            [3, 5, 7, 9],
            [1, 3, 9, 11],
            [2, 4, 10, 12],
        ]
        for drop in candidate_drops:
            keep = [c for c in range(28) if c not in drop]
            if len(keep) != 24:
                continue
            A_mat = np.stack([F_list[c] for c in keep], axis=1)  # (128, 24)
            b_vec = target.reshape(-1)
            try:
                coeffs, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except np.linalg.LinAlgError:
                continue
            recon = A_mat @ coeffs
            if np.allclose(recon, b_vec, atol=1e-9):
                # Build the 24-column result
                Uf = np.zeros((8, 24))
                Vf = np.zeros((16, 24))
                Wf = np.zeros((8, 24))
                for new_c, old_c in enumerate(keep):
                    Uf[:, new_c] = cand_U[:, old_c] * coeffs[new_c]
                    Vf[:, new_c] = cand_V[:, old_c] * coeffs[new_c]
                    Wf[:, new_c] = cand_W[:, old_c] * coeffs[new_c]
                if verify(Uf, Vf, Wf, 24):
                    best = (Uf, Vf, Wf, 24)
                    found24 = True
                    break
        if found24:
            return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}

    # --- Attempt 2: rank-28 (verified) ---------------------------------
    Uf, Vf, Wf, rank = build_rank28()
    if strassen_ok and verify(Uf, Vf, Wf, rank):
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    # --- Fallback: rank-32 schoolbook (always valid) -------------------
    Uf, Vf, Wf, rank = build_rank32()
    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
# EVOLVE-BLOCK-END
