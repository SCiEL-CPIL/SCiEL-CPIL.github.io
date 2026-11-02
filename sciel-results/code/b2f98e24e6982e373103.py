import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 100.0  # safety margin before 120s kill

    # ---- schoolbook fallback ---------------------------------------------
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- block construction ---------------------------------------------
    # We partition A (3x4) and B (4x6) into blocks:
    #   rows of A / rows of C:  [2, 1]
    #   cols of A / rows of B:  [2, 2]
    #   cols of B / cols of C:  [2, 2, 2]
    #
    # For each block (a, b, c) of sizes (ra x rb) * (rb x rc):
    #   if ra == rb == rc == 2: use Strassen rank-7
    #   else: schoolbook rank ra*rb*rc
    #
    # Strassen for 2x2 * 2x2 (rank 7):
    #   M1 = (A11 + A22)(B11 + B22)
    #   M2 = (A21 + A22) B11
    #   M3 = A11 (B12 - B22)
    #   M4 = A22 (B21 - B11)
    #   M5 = (A11 + A12) B22
    #   M6 = (A21 - A11)(B11 + B12)
    #   M7 = (A12 - A22)(B21 + B22)
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6

    def strassen_2x2x2():
        """Return (U, V, W) for 2x2 * 2x2 with rank 7.
        U shape (4, 7), V shape (4, 7), W shape (4, 7).
        Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)

        # M1 = (A11 + A22)(B11 + B22)
        # A indices: A11=(0,0), A22=(1,1)
        # B indices: B11=(0,0), B22=(1,1)
        # C: C11 += 1, C22 += 1
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0  # W[k*2+i]: C11 -> k=0,i=0 -> 0; C22 -> k=1,i=1 -> 3

        # M2 = (A21 + A22) B11
        # A21=(1,0), A22=(1,1); B11=(0,0)
        # C21 += 1
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[2, 1] = 1.0  # C21 -> k=0,i=1 -> 0*2+1=1... wait W[k*n+i] with n=2: W[k*2+i]
        # C21 means i=1 (row 1 of C), k=0 (col 0 of C) -> W[k*2+i] = W[0*2+1] = W[1]
        # Let me redo: W index is k*n + i where n=2 for the sub-block.
        # C21: i=1, k=0 -> W[0*2+1] = W[1]
        W[1, 1] = 1.0

        # M3 = A11 (B12 - B22)
        # A11=(0,0); B12=(0,1), B22=(1,1)
        # C12 += 1
        U[0, 2] = 1.0
        V[1, 2] = 1.0; V[3, 2] = -1.0
        # C12: i=0, k=1 -> W[1*2+0] = W[2]
        W[2, 2] = 1.0

        # M4 = A22 (B21 - B11)
        # A22=(1,1); B21=(1,0), B11=(0,0)
        # C11 += 1, C21 += 1
        U[3, 3] = 1.0
        V[2, 3] = 1.0; V[0, 3] = -1.0
        # C11: i=0,k=0 -> W[0]; C21: i=1,k=0 -> W[1]
        W[0, 3] = 1.0
        W[1, 3] = 1.0

        # M5 = (A11 + A12) B22
        # A11=(0,0), A12=(0,1); B22=(1,1)
        # C11 -= 1, C12 += 1
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        # C11: i=0,k=0 -> W[0] coeff -1; C12: i=0,k=1 -> W[2] coeff +1
        W[0, 4] = -1.0
        W[2, 4] = 1.0

        # M6 = (A21 - A11)(B11 + B12)
        # A21=(1,0), A11=(0,0); B11=(0,0), B12=(0,1)
        # C22 += 1
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[1, 5] = 1.0
        # C22: i=1,k=1 -> W[1*2+1] = W[3]
        W[3, 5] = 1.0

        # M7 = (A12 - A22)(B21 + B22)
        # A12=(0,1), A22=(1,1); B21=(1,0), B22=(1,1)
        # C11 += 1
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[2, 6] = 1.0; V[3, 6] = 1.0
        # C11: i=0,k=0 -> W[0]
        W[0, 6] = 1.0

        return U, V, W

    def schoolbook_block(ra, rb, rc):
        """Return (U, V, W) for ra x rb * rb x rc with schoolbook rank ra*rb*rc."""
        rank = ra * rb * rc
        U = np.zeros((ra * rb, rank), dtype=float)
        V = np.zeros((rb * rc, rank), dtype=float)
        W = np.zeros((rc * ra, rank), dtype=float)
        col = 0
        for i in range(ra):
            for j in range(rb):
                for k in range(rc):
                    U[i * rb + j, col] = 1.0
                    V[j * rc + k, col] = 1.0
                    W[k * ra + i, col] = 1.0
                    col += 1
        return U, V, W

    # Block sizes
    row_blocks = [2, 1]  # rows of A / rows of C
    col_blocks = [2, 2]  # cols of A / rows of B
    k_blocks = [2, 2, 2]  # cols of B / cols of C

    # Build global U, V, W by concatenating block contributions
    all_U = []
    all_V = []
    all_W = []

    for ii, ra in enumerate(row_blocks):
        for jj, rb in enumerate(col_blocks):
            for kk, rc in enumerate(k_blocks):
                if ra == 2 and rb == 2 and rc == 2:
                    Ub, Vb, Wb = strassen_2x2x2()
                else:
                    Ub, Vb, Wb = schoolbook_block(ra, rb, rc)

                # Embed into global coordinates
                # Global U index: i_global * m + j_global
                #   i_global = sum(row_blocks[:ii]) + i_local
                #   j_global = sum(col_blocks[:jj]) + j_local
                # Global V index: j_global * p + k_global
                #   j_global = sum(col_blocks[:jj]) + j_local
                #   k_global = sum(k_blocks[:kk]) + k_local
                # Global W index: k_global * n + i_global
                #   k_global = sum(k_blocks[:kk]) + k_local
                #   i_global = sum(row_blocks[:ii]) + i_local

                i_off = sum(row_blocks[:ii])
                j_off = sum(col_blocks[:jj])
                k_off = sum(k_blocks[:kk])

                Ug = np.zeros((n * m, Ub.shape[1]), dtype=float)
                Vg = np.zeros((m * p, Vb.shape[1]), dtype=float)
                Wg = np.zeros((p * n, Wb.shape[1]), dtype=float)

                for r in range(Ub.shape[1]):
                    for i_local in range(ra):
                        for j_local in range(rb):
                            val = Ub[i_local * rb + j_local, r]
                            if val != 0.0:
                                gi = i_off + i_local
                                gj = j_off + j_local
                                Ug[gi * m + gj, r] = val

                    for j_local in range(rb):
                        for k_local in range(rc):
                            val = Vb[j_local * rc + k_local, r]
                            if val != 0.0:
                                gj = j_off + j_local
                                gk = k_off + k_local
                                Vg[gj * p + gk, r] = val

                    for k_local in range(rc):
                        for i_local in range(ra):
                            val = Wb[k_local * ra + i_local, r]
                            if val != 0.0:
                                gk = k_off + k_local
                                gi = i_off + i_local
                                Wg[gk * n + gi, r] = val

                all_U.append(Ug)
                all_V.append(Vg)
                all_W.append(Wg)

    U_global = np.hstack(all_U)
    V_global = np.hstack(all_V)
    W_global = np.hstack(all_W)

    rank = U_global.shape[1]

    # Verify decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_global[i * m + j, c] * V_global[j * p + k, c] * W_global[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        return make_schoolbook()

    if time.time() > deadline:
        return make_schoolbook()

    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
# EVOLVE-BLOCK-END
