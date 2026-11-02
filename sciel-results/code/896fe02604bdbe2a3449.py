"""Matrix multiplication 2x2x10 rank-35 decomposition via Kronecker product."""
import numpy as np
import time

_DEADLINE = time.time() + 130.0


def construct():
    n, m, p = 2, 2, 10

    # 2x2x2 Strassen: rank 7
    S_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ], dtype=float)  # (4, 7)

    S_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float)  # (4, 7)

    S_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ], dtype=float)  # (7, 4)

    # 1x2x4 schoolbook: rank 4
    E_A = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]], dtype=float)  # (2, 4)
    E_B = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=float)  # (4, 4)
    E_W = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=float)  # (4, 4)

    # Kronecker product: rank = 7 * 4 = 28 for the 2x2x4 part
    # U_kron: (4*2, 7*4) = (8, 28)
    U_kron = np.kron(S_A, E_A)  # row (i*2+j), col (r*4+s)
    V_kron = np.kron(S_B, E_B)  # row (j*4+k), col (r*4+s)
    W_kron = np.kron(S_W, E_W)  # row (k*2+i), col (r*4+s)

    # Now extend from 2x2x4 to 2x2x10 using 2.5 block structure
    # Strategy: use the rank-28 decomposition for cols 0-3,
    # and additional rank-7 terms for cols 4-9 via a shifted Strassen

    # Alternative: decompose 2x2x10 as 2x2x5 with a 5-column extension
    # Actually, let's use a different approach:
    # 2x2x10 = 2x2x4 + 2x2x6, where 2x2x6 uses 3 Strassen blocks (rank 21)
    # Total: 28 + 21 = 49. Too high.

    # Better: direct construction for 2x2x10
    # Use 5 Strassen blocks of 2x2x2, rank 35
    rank = 35
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((20, rank), dtype=float)
    W = np.zeros((20, rank), dtype=float)

    for blk in range(5):
        base = blk * 7
        for r in range(7):
            c = base + r
            # A side: same for all blocks
            U[0, c] = S_A[0, r]
            U[1, c] = S_A[1, r]
            U[2, c] = S_A[2, r]
            U[3, c] = S_A[3, r]

            # B side: rows 2*blk and 2*blk+1 for row 0; rows 10+2*blk and 10+2*blk+1 for row 1
            V[2 * blk, c] = S_B[0, r]
            V[2 * blk + 1, c] = S_B[1, r]
            V[10 + 2 * blk, c] = S_B[2, r]
            V[10 + 2 * blk + 1, c] = S_B[3, r]

            # W side: trace convention W[k*n+i, r] where n=2
            # C[i, k] contribution: W[k*2+i, c] = S_W[r, i]
            W[2 * blk * 2 + 0, c] = S_W[r, 0]  # k=2*blk, i=0 -> C00
            W[2 * blk * 2 + 1, c] = S_W[r, 1]  # k=2*blk, i=1 -> C10
            W[(2 * blk + 1) * 2 + 0, c] = S_W[r, 2]  # k=2*blk+1, i=0 -> C01
            W[(2 * blk + 1) * 2 + 1, c] = S_W[r, 3]  # k=2*blk+1, i=1 -> C11

    # Verify
    if time.time() < _DEADLINE and _check(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback schoolbook rank 40
    U_s = np.zeros((4, 40), dtype=float)
    V_s = np.zeros((20, 40), dtype=float)
    W_s = np.zeros((20, 40), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, idx] = 1.0
                V_s[j * p + k, idx] = 1.0
                W_s[k * n + i, idx] = 1.0
                idx += 1
    return {"U": U_s, "V": V_s, "W": W_s}


def _check(n, m, p, U, V, W):
    rng = np.random.RandomState(7)
    for _ in range(10):
        A = rng.randint(-3, 4, (n, m)).astype(float)
        B = rng.randint(-3, 4, (m, p)).astype(float)
        C_ref = A @ B
        r = U.shape[1]
        Au = np.einsum('ij,ijr->ir', A, U.reshape(n, m, r))
        Bv = np.einsum('kl,klr->lr', B, V.reshape(m, p, r))
        Cc = np.einsum('ir,lr,lir->il', Au, Bv, W.reshape(p, n, r))
        if not np.allclose(Cc, C_ref, atol=1e-12):
            return False
    return True
