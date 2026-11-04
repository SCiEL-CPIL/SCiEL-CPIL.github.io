"""Rewritten program: rank-50 decomposition via Strassen blocks with a different structural layout."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 7

    # We build a rank-50 decomposition using 6 Strassen blocks (2x2x2, rank 7 each = 42)
    # plus 8 rank-1 terms for the last column of B.
    #
    # Strategy:
    #   - B has 7 columns. We split into: 4 cols handled by 4 Strassen blocks,
    #     2 cols by 2 Strassen blocks, 1 col by 8 rank-1 terms.
    #   - Each Strassen block covers a 2x2 submatrix of A (rows 0-1) and a 2x2
    #     sub-block of B (2 rows x 2 cols), producing a 2x2 sub-block of C.
    #
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (28, rank)
    # W shape: (p*n, rank) = (14, rank)

    rank = 50
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    def _strassen(i0, j0, k0):
        """Add one 2x2x2 Strassen block.
        i0: starting row of A block (0 or 1, but we only have 2 rows so i0=0 always for 2x2)
        j0: starting col of A / row of B
        k0: starting col of B
        """
        nonlocal col
        # A indices: row in {i0, i0+1}, col in {j0, j0+1}
        # B indices: row in {j0, j0+1}, col in {k0, k0+1}
        # C indices: row in {k0, k0+1}, col in {i0, i0+1}  -> W row = k*n + i

        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1

        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1

        # M3 = A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1

        # M4 = A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1

        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1

        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1

        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 6 Strassen blocks covering B columns 0..5
    # Block layout: (j0, k0)
    blocks = [
        (0, 0),  # A cols 0-1, B rows 0-1, B cols 0-1 -> C rows 0-1, cols 0-1
        (0, 2),  # A cols 0-1, B rows 0-1, B cols 2-3 -> C rows 2-3, cols 0-1
        (2, 0),  # A cols 2-3, B rows 2-3, B cols 0-1 -> C rows 0-1, cols 0-1
        (2, 2),  # A cols 2-3, B rows 2-3, B cols 2-3 -> C rows 2-3, cols 0-1
        (0, 4),  # A cols 0-1, B rows 0-1, B cols 4-5 -> C rows 4-5, cols 0-1
        (2, 4),  # A cols 2-3, B rows 2-3, B cols 4-5 -> C rows 4-5, cols 0-1
    ]

    for (j0, k0) in blocks:
        _strassen(0, j0, k0)

    # 8 rank-1 terms for B column 6 (the 7th column)
    for i in range(n):
        for j in range(m):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1

    # Verify
    if time.time() - t0 < 10:
        Uf = U.astype(float)
        Vf = V.astype(float)
        Wf = W.astype(float)
        # Check: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += Uf[i*m+j, r] * Vf[j*p+k, r] * Wf[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            # Fallback: return schoolbook rank-56 if verification fails
            rank_fb = n * m * p
            U_fb = np.zeros((n*m, rank_fb), dtype=int)
            V_fb = np.zeros((m*p, rank_fb), dtype=int)
            W_fb = np.zeros((p*n, rank_fb), dtype=int)
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, c] = 1
                        V_fb[j*p+k, c] = 1
                        W_fb[k*n+i, c] = 1
                        c += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
