import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 144.0  # 6s safety margin before 150s kill

# Strassen 2x2x2 coefficient sets: index by flat (row*2+col)
# alpha: A-coeffs, beta: B-coeffs, recon: list of (a_row, b_col, gamma)
STRASSEN = [
    ( [1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)] ),
    ( [0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)] ),
    ( [1,0,0,0], [0,1,0,-1],[(0,1,1),(1,1,1)] ),
    ( [0,0,0,1], [-1,0,1,0],[(0,0,1),(1,0,1)] ),
    ( [1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)] ),
    ( [-1,0,1,0],[1,1,0,0], [(1,1,1)] ),
    ( [0,1,0,-1],[0,0,1,1], [(0,0,1)] ),
]


def _build_rank77():
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    col = 0
    for s in range(7):
        as_, bs_, rs_ = STRASSEN[s]
        # Part 1: inner Strassen for the two 2x2 blocks (result cols 0,1)
        for si in range(7):
            ai_, bi_, ri_ = STRASSEN[si]
            # U: row index = i*m + k ; i in 0..3, k in 0..3
            for i in range(4):
                for k in range(4):
                    ar, br = i // 2, i % 2
                    xc = k % 2
                    outer = as_[ar*2+br]
                    inner = ai_[xc]
                    U[i*4+k, col] += outer * inner
            # V: row index = k*p + j ; k in 0..3, j in 0..5
            for k in range(4):
                for j in range(6):
                    ac, bc = k // 2, k % 2
                    yr, yc = j // 3, j % 3
                    if yc >= 2:
                        continue
                    outer = bs_[ac*2+bc]
                    inner = bi_[yr*2+yc]
                    V[k*6+j, col] += outer * inner
            # W: row index = j*n + i ; j in 0..5, i in 0..3
            for (a_row, b_col, gamma) in rs_:
                for (zr, zc, zgamma) in ri_:
                    i = a_row*2 + zr
                    j = b_col*3 + zc
                    W[j*4+i, col] += gamma * zgamma
            col += 1

        # Part 2: schoolbook for the third column (result col 2), 4 scalar mults
        for u in range(2):
            for v in range(2):
                # U: i in 0..3, k in 0..3 ; inner fixed to v on xcol
                for i in range(4):
                    for k in range(4):
                        ar, br = i // 2, i % 2
                        if k % 2 != v:
                            continue
                        U[i*4+k, col] += as_[ar*2+br]
                # V: k in 0..3, result col 2 -> j=2
                for k in range(4):
                    ac, bc = k // 2, k % 2
                    V[k*6+2, col] += bs_[ac*2+bc]
                # W: j=2 fixed, i = u ; outer recon gamma with b_col=0, a_row = u//2
                a_row = u // 2
                for (ar2, bc2, gamma) in rs_:
                    if ar2 == a_row and bc2 == 0:
                        W[2*4+u, col] += gamma
                col += 1

    assert col == R, col
    return U, V, W


def _verify_decomposition(U, V, W, n=4, m=4, p=6):
    rng = np.random.RandomState(1234)
    for _ in range(2):
        A = rng.randint(-5, 5, size=(n, m))
        B = rng.randint(-5, 5, size=(m, p))
        AB = A @ B
        result = np.zeros((n, p), dtype=float)
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    result[i, j] += np.sum(U[i*m+k, :] * V[k*p+j, :] * W[j*n+i, :])
        if not np.allclose(result, AB, atol=1e-6):
            return False
    return True


def _try_rank_reduce(U, V, W, n=4, m=4, p=6):
    """Remove columns that are near-zero; coordinate-descent LS not guaranteed, so
    only drop provably-zero columns and re-verify."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    R = U.shape[1]
    deadline = START_TIME + TIME_LIMIT

    for _ in range(20):
        if deadline - time.time() < 15:
            break
        improved = False
        for r in range(R):
            if deadline - time.time() < 18:
                break
            if (np.linalg.norm(U[:, r]) < 1e-9 and np.linalg.norm(V[:, r]) < 1e-9
                    and np.linalg.norm(W[:, r]) < 1e-9):
                U = np.column_stack([U[:, j] for j in range(R) if j != r])
                V = np.column_stack([V[:, j] for j in range(R) if j != r])
                W = np.column_stack([W[:, j] for j in range(R) if j != r])
                R -= 1
                improved = True
                break
        if not improved:
            break
    return U, V, W, R


def construct():
    U, V, W = _build_rank77()

    if not _verify_decomposition(U, V, W):
        # Guaranteed-valid fallback: full schoolbook (one rank-1 slice per (i,k,j))
        n, m, p = 4, 4, 6
        R0 = n * m * p
        U0 = np.zeros((n*m, R0))
        V0 = np.zeros((m*p, R0))
        W0 = np.zeros((p*n, R0))
        c = 0
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    U0[i*m+k, c] = 1.0
                    V0[k*p+j, c] = 1.0
                    W0[j*n+i, c] = 1.0
                    c += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    if time.time() - START_TIME < TIME_LIMIT - 25:
        U2, V2, W2, _ = _try_rank_reduce(U, V, W)
        if _verify_decomposition(U2, V2, W2):
            U, V, W = U2, V2, W2

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
