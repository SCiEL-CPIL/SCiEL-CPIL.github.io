import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 150.0

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded(margin=5.0):
    return _elapsed() > _DEADLINE - margin


def _schoolbook(n, m, p):
    """Guaranteed-correct schoolbook decomposition, rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = (i * m + j) * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
    return U, V, W


def _verify(U, V, W, n, m, p, trials=8):
    """Check exactness of the decomposition on random inputs."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        # C[i,k] = sum_t U[i*m+j? no] ...  C = W^T (V^T B) (U^T A)
        # U shape (n*m, rank): Uvec = U.T @ A.flatten()
        uvec = U.T @ A.reshape(n * m)
        vvec = V.T @ B.reshape(m * p)
        # combine: for each t, (uvec[t] * vvec[t]) outer ... C = sum_t uvec[t]*vvec[t] -> W col
        # C = W.T @ (uvec * vvec)
        C_rec = W.T @ (uvec * vvec)
        if not np.allclose(C_rec, C_true, atol=1e-8, rtol=1e-8):
            return False
    return True


def _strassen_block(U, V, W, uA, vB, wC, col):
    """
    Apply Strassen rank-7 to a 2x2x2 block.
    uA[i][j] -> U index of A[i,j]; vB[j][k] -> V index of B[j,k]; wC[i][k] -> W index of C[i,k].
    Returns new col.
    """
    a, b, c, d = uA
    e, f, g, h = vB
    c00, c01, c10, c11 = wC

    def term(u, v, w):
        nonlocal col
        for idx, val in u:
            U[idx, col] += val
        for idx, val in v:
            V[idx, col] += val
        for idx, val in w:
            W[idx, col] += val
        col += 1

    # M1 = (a+d)(e+h)
    term([(a, 1), (d, 1)], [(e, 1), (h, 1)], [(c00, 1), (c11, 1)])
    # M2 = (c+d)(e)
    term([(c, 1), (d, 1)], [(e, 1)], [(c10, 1), (c11, -1)])
    # M3 = (a)(f-h)
    term([(a, 1)], [(f, 1), (h, -1)], [(c01, 1), (c11, 1)])
    # M4 = (d)(g-e)
    term([(d, 1)], [(g, 1), (e, -1)], [(c00, 1), (c10, 1)])
    # M5 = (a+b)(h)
    term([(a, 1), (b, 1)], [(h, 1)], [(c00, -1), (c01, 1)])
    # M6 = (c-a)(e+f)
    term([(c, 1), (a, -1)], [(e, 1), (f, 1)], [(c11, 1)])
    # M7 = (b-d)(g+h)
    term([(b, 1), (d, -1)], [(g, 1), (h, 1)], [(c00, 1)])
    return col


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5

    if _deadline_exceeded():
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(i, k):
        return k * n + i

    # Total rank: 7+7+11+11+6+6 = 48
    rank = 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    # Block 1: A11(2x2) * B11(2x2) -> C11  (i in {0,1}, j in {0,1}, k in {0,1})
    col = _strassen_block(
        U, V, W,
        uA=[[u_idx(0, 0), u_idx(0, 1)], [u_idx(1, 0), u_idx(1, 1)]],
        vB=[[v_idx(0, 0), v_idx(0, 1)], [v_idx(1, 0), v_idx(1, 1)]],
        wC=[[w_idx(0, 0), w_idx(0, 1)], [w_idx(1, 0), w_idx(1, 1)]],
        col=col,
    )

    # Block 2: A12(2x2) * B21(2x2) -> C11  (A cols {2,3}, B rows {2,3}, k in {0,1})
    col = _strassen_block(
        U, V, W,
        uA=[[u_idx(0, 2), u_idx(0, 3)], [u_idx(1, 2), u_idx(1, 3)]],
        vB=[[v_idx(2, 0), v_idx(2, 1)], [v_idx(3, 0), v_idx(3, 1)]],
        wC=[[w_idx(0, 0), w_idx(0, 1)], [w_idx(1, 0), w_idx(1, 1)]],
        col=col,
    )

    # Block 3: A11(2x2) * B12(2x3) -> C12
    #   part A: k in {2,3} Strassen
    col = _strassen_block(
        U, V, W,
        uA=[[u_idx(0, 0), u_idx(0, 1)], [u_idx(1, 0), u_idx(1, 1)]],
        vB=[[v_idx(0, 2), v_idx(0, 3)], [v_idx(1, 2), v_idx(1, 3)]],
        wC=[[w_idx(0, 2), w_idx(0, 3)], [w_idx(1, 2), w_idx(1, 3)]],
        col=col,
    )
    #   part B: k=4 schoolbook (2x2x1)
    for i in range(2):
        for j in range(2):
            U[u_idx(i, j), col] = 1.0
            V[v_idx(j, 4), col] = 1.0
            W[w_idx(i, 4), col] = 1.0
            col += 1

    # Block 4: A12(2x2) * B22(2x3) -> C12
    #   part A: k in {2,3} Strassen
    col = _strassen_block(
        U, V, W,
        uA=[[u_idx(0, 2), u_idx(0, 3)], [u_idx(1, 2), u_idx(1, 3)]],
        vB=[[v_idx(2, 2), v_idx(2, 3)], [v_idx(3, 2), v_idx(3, 3)]],
        wC=[[w_idx(0, 2), w_idx(0, 3)], [w_idx(1, 2), w_idx(1, 3)]],
        col=col,
    )
    #   part B: k=4 schoolbook (2x2x1)
    for i in range(2):
        for j in range(2, 4):
            U[u_idx(i, j), col] = 1.0
            V[v_idx(j, 4), col] = 1.0
            W[w_idx(i, 4), col] = 1.0
            col += 1

    # Block 5: A31(1x2) * B13(2x3) -> C31  schoolbook (1x2x3)
    for j in range(2):
        for k in range(3):
            U[u_idx(2, j), col] = 1.0
            V[v_idx(j, k), col] = 1.0
            W[w_idx(2, k), col] = 1.0
            col += 1

    # Block 6: A32(1x2) * B23(2x3) -> C31  schoolbook (1x2x3)
    for j in range(2, 4):
        for k in range(3):
            U[u_idx(2, j), col] = 1.0
            V[v_idx(j, k), col] = 1.0
            W[w_idx(2, k), col] = 1.0
            col += 1

    assert col == rank, f"col {col} != rank {rank}"

    # Verify the decomposition; fall back to schoolbook if it fails.
    if _verify(U, V, W, n, m, p, trials=10):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: guaranteed correct
    if _deadline_exceeded():
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
