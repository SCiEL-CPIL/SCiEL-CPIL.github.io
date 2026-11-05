"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock limit

    n, m, p = DIMENSIONS  # 3, 3, 6

    def verify(U, V, W):
        """Verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        """Trivial decomposition with rank = n*m*p = 54 (guaranteed valid)."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Construct a rank-23 decomposition for 3x3 * 3x3 ----
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # We build 23 scalar products. The first 15 are "diagonal" (one j each),
    # the last 8 are "Strassen-like" cross terms (two j's each).
    #
    # Diagonal term (i,j,k): contributes A[i,j]*B[j,k]*delta_{i,k} to C[i,k].
    # Cross term (i,j1,j2,k): contributes (A[i,j1]+A[i,j2])*(B[j1,k]+B[j2,k])*delta_{i,k}.
    #
    # For each (i,k) we need exactly one diagonal term (its own j=k) plus an
    # odd number of cross terms (so the net delta_{i,k} coefficient is 1).
    # With 8 cross terms, the diagonal (i,k=i) terms each need 0 cross terms
    # (odd = 1 diagonal) and off-diagonal (i,k) need 1 cross term (odd total).
    # We use 4 cross terms, each covering a pair of off-diagonal (i,k) entries
    # with i != k and i != j1, i != j2.

    def build_3x3_23():
        """Return (U, V, W) each a list of 23 vectors for 3x3*3x3.
        U[i*3+j], V[j*3+k], W[k*3+i]."""
        U = {}  # (i,j) -> list of 23 coeffs
        V = {}
        W = {}
        for i in range(3):
            for j in range(3):
                U[(i, j)] = [0] * 23
                V[(j, i)] = [0] * 23  # placeholder, fill below
        # Initialize all
        for i in range(3):
            for j in range(3):
                U[(i, j)] = [0] * 23
        for j in range(3):
            for k in range(3):
                V[(j, k)] = [0] * 23
        for k in range(3):
            for i in range(3):
                W[(k, i)] = [0] * 23

        # 15 diagonal terms (indices 0..14): (i,j,k) with i==k
        idx = 0
        for i in range(3):
            for j in range(3):
                k = i
                U[(i, j)][idx] = 1
                V[(j, k)][idx] = 1
                W[(k, i)][idx] = 1
                idx += 1

        # 4 cross terms (indices 15..18) covering off-diagonal (i,k) pairs.
        # Off-diagonal (i,k) pairs with i != k: (0,1),(0,2),(1,0),(1,2),(2,0),(2,1) -> 6 pairs.
        # Each cross term covers 2 pairs. We need each off-diagonal pair covered exactly once.
        # A cross term (i, j1, j2, k) with i != k covers:
        #   - the (i,k) diagonal contribution (we want this to be 1 for off-diagonal i!=k)
        # Actually each cross term is associated with a specific (i,k) where i!=k.
        # We have 6 off-diagonal (i,k) but only 4 cross terms. So each cross term must
        # cover a pair of (i,k) entries. But a single cross term (i,j1,j2,k) only has one
        # fixed (i,k). So we can't cover two (i,k) with one cross term.

        # Reconsider: we need 6 off-diagonal (i,k) entries each to have odd cross coverage,
        # and 3 diagonal (i,k) entries to have even (0) cross coverage.
        # With 4 cross terms, total cross "slots" = 4, but we need 6 off-diagonal covered.
        # Contradiction. So we need at least 6 cross terms if each covers one (i,k).
        #
        # Let me use 6 cross terms instead -> total 15 + 6 = 21 terms.
        # Actually let me recount: we want total rank R. Diagonal = 9 (i==k, one per (i,k)).
        # Wait, diagonal terms: for each (i,k) with i==k, we need sum over j of one term per j?
        # No. Let me restart the counting properly.

        # PROPER COUNTING:
        # For each (i,k) we need: sum_{j} [diag_j(i,k)] + sum_{cross} [cross_contrib] = 1,
        # where diag_j(i,k) = 1 if there's a diagonal term with this (i,k) and that j.
        # A diagonal term is indexed by (i,j,k) and contributes to C[i,k] only (delta_{i,k} baked in W).
        # For the (i,k) with i==k: we can use ALL j's as separate diagonal terms, OR fewer.
        # Simplest: for i==k, use 1 diagonal term (any j) -> contributes 1. Need 0 cross.
        # For i!=k: use 0 diagonal terms, need 1 cross term.
        # Number of (i,k) with i==k: 3. Number with i!=k: 6.
        # Diagonal terms: 3 (one per i==k). Cross terms: 6 (one per i!=k).
        # Total: 3 + 6 = 9? That seems too low. Let me check the diagonal term for i==k.

        # A diagonal term (i, j, k=i) contributes A[i,j]*B[j,i] to C[i,i].
        # But C[i,i] = sum_{j'} A[i,j']*B[j',i]. We only get ONE j' (the chosen j) from the
        # diagonal term. We need ALL j'. So we need 3 diagonal terms per (i,i) (one per j),
        # OR use cross terms to supply the missing j's.

        # So for (i,i): need 3 terms covering j=0,1,2. Use 3 diagonal terms (i,0,i),(i,1,i),(i,2,i).
        # That's 3*3 = 9 diagonal terms total (for the 3 diagonal (i,i) positions).
        # For (i,k) i!=k: need 1 cross term each -> 6 cross terms.
        # Total: 9 + 6 = 15. That's rank 15! Let me verify this works.

        # Let me rebuild with this cleaner scheme: 9 diagonal + 6 cross = 15 terms.
        U = {}
        V = {}
        W = {}
        for i in range(3):
            for j in range(3):
                U[(i, j)] = [0] * 15
        for j in range(3):
            for k in range(3):
                V[(j, k)] = [0] * 15
        for k in range(3):
            for i in range(3):
                W[(k, i)] = [0] * 15

        # 9 diagonal terms (indices 0..8): (i, j, i) for all i, j
        idx = 0
        for i in range(3):
            for j in range(3):
                k = i
                U[(i, j)][idx] = 1
                V[(j, k)][idx] = 1
                W[(k, i)][idx] = 1
                idx += 1

        # 6 cross terms (indices 9..14): (i, j1, j2, k) for i != k
        # Each cross term: (A[i,j1]+A[i,j2])*(B[j1,k]+B[j2,k]) contributes to C[i,k] (i!=k).
        # We need exactly one cross term per (i,k) with i!=k.
        # For a given (i,k) with i!=k, choose j1, j2 distinct and both != i (so they are the
        # two "other" indices). There's a unique pair {0,1,2}\{i}. Use j1, j2 = that pair.
        idx = 9
        for i in range(3):
            for k in range(3):
                if i == k:
                    continue
                others = [x for x in range(3) if x != i]
                j1, j2 = others[0], others[1]
                U[(i, j1)][idx] = 1
                U[(i, j2)][idx] = 1
                V[(j1, k)][idx] = 1
                V[(j2, k)][idx] = 1
                W[(k, i)][idx] = 1
                idx += 1

        # Now convert to arrays: U[i*3+j], V[j*3+k], W[k*3+i]
        Ua = np.zeros((9, 15))
        Va = np.zeros((9, 15))
        Wa = np.zeros((9, 15))
        for i in range(3):
            for j in range(3):
                Ua[i*3+j] = U[(i, j)]
        for j in range(3):
            for k in range(3):
                Va[j*3+k] = V[(j, k)]
        for k in range(3):
            for i in range(3):
                Wa[k*3+i] = W[(k, i)]
        return Ua, Va, Wa

    # Verify the 3x3*3x3 rank-15 decomposition
    Ua, Va, Wa = build_3x3_23()
    ok_3x3 = True
    for i in range(3):
        for k in range(3):
            # C[i,k] = sum_j sum_r Ua[i*3+j, r]*Va[j*3+k, r]*Wa[k*3+i, r]
            val = 0.0
            for j in range(3):
                val += np.dot(Ua[i*3+j] * Va[j*3+k], Wa[k*3+i])
            if abs(val - 1.0) > 1e-9:
                ok_3x3 = False
                break
        if not ok_3x3:
            break

    if ok_3x3:
        # Build the 3x3 * 3x6 decomposition by reusing the rank-15 3x3*3x3 twice.
        # B is 3x6 = [B1 | B2], each 3x3. C = [A*B1 | A*B2].
        # Rank = 15 + 15 = 30.
        # U: 9 x 30, V: 18 x 30, W: 18 x 30.
        # First 15 columns: B1 (k=0,1,2). Last 15 columns: B2 (k=3,4,5).
        R = 30
        U_full = np.zeros((9, R))
        V_full = np.zeros((18, R))
        W_full = np.zeros((18, R))

        # Half 1: k = 0,1,2 (B1). Maps to the 3x3*3x3 with k' = k.
        for i in range(3):
            for j in range(3):
                U_full[i*3+j, 0:15] = Ua[i*3+j]
        for j in range(3):
            for kp in range(3):  # k' in 0..2
                V_full[j*6+kp, 0:15] = Va[j*3+kp]
        for kp in range(3):
            for i in range(3):
                W_full[kp*3+i, 0:15] = Wa[kp*3+i]

        # Half 2: k = 3,4,5 (B2). Maps to the 3x3*3x3 with k' = k-3.
        for i in range(3):
            for j in range(3):
                U_full[i*3+j, 15:30] = Ua[i*3+j]
        for j in range(3):
            for kp in range(3):  # k' in 0..2, actual k = kp+3
                V_full[j*6+(kp+3), 15:30] = Va[j*3+kp]
        for kp in range(3):
            for i in range(3):
                W_full[(kp+3)*3+i, 15:30] = Wa[kp*3+i]

        # Verify full decomposition
        if verify(U_full.tolist(), V_full.tolist(), W_full.tolist()):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: schoolbook (guaranteed valid, rank 54)
    return make_schoolbook()

# EVOLVE-BLOCK-END
