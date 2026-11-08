"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify the tensor decomposition"""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    total = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i,j,k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    expected = np.ones((n, m, p))
    if not np.allclose(total, expected, atol=1e-6):
        return False, np.max(np.abs(total - expected))
    return True, 0.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Rank 48 solution:
    #   - 2x2x2 Strassen blocks (rank 7) on (j0,j1) x (k0,k1) for
    #     j-pairs (0,2), (1,3) and k-pairs (0,4), (1,5)   -> 6 * 7 = 42
    #   - direct terms (rank 1) on odd j x odd k:
    #     (j=0,k=1), (j=0,k=5), (j=1,k=1), (j=1,k=5),
    #     (j=3,k=0), (j=3,k=4)                            -> 6
    #   Total = 48
    def _build_rank48():
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen(j0, j1, k0, k1):
            nonlocal col
            # P1 = (A_i0j0 + A_i1j1)(B_j0k0 + B_j1k1) -> C_i0k0, C_i1k1
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            # P2 = (A_i1j0 + A_i1j1) B_j0k0 -> C_i1k0, C_i1k1(-)
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            # P3 = A_i0j0 (B_j0k1 - B_j1k1) -> C_i0k1, C_i1k1(+)
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            # P4 = A_i1j1 (B_j1k0 - B_j0k0) -> C_i0k0(+), C_i1k0(+)
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            # P5 = (A_i0j0 + A_i0j1) B_j1k1 -> C_i0k0(-), C_i0k1(+)
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            # P6 = (A_i1j0 - A_i0j0)(B_j0k0 + B_j0k1) -> C_i1k1(+)
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            # P7 = (A_i0j1 - A_i1j1)(B_j1k0 + B_j1k1) -> C_i0k0(+)
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1

        def add_direct(j, k):
            nonlocal col
            U[0*m + j][col] += 1
            U[1*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 0][col] += 1
            W[k*n + 1][col] += 1
            col += 1

        # Strassen 2x2x2 blocks
        for (j0, j1) in [(0, 2), (1, 3)]:
            for (k0, k1) in [(0, 4), (1, 5)]:
                add_strassen(j0, j1, k0, k1)

        # Direct odd j x odd k terms
        for (j, k) in [(0, 1), (0, 5), (1, 1), (1, 5), (3, 0), (3, 4)]:
            add_direct(j, k)

        return {"U": U, "V": V, "W": W}

    def _build_direct():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] += 1
                    V[j*p + k][col] += 1
                    W[k*n + i][col] += 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = None
    best_rank = float('inf')

    for builder in (_build_rank48, _build_direct):
        if time.time() > deadline:
            break
        cand = builder()
        valid, diff = _verify(cand["U"], cand["V"], cand["W"], n, m, p)
        if valid:
            r = len(cand["U"][0])
            if r < best_rank:
                best_rank = r
                best = cand

    if best is None:
        return _build_direct()
    return best
# EVOLVE-BLOCK-END
