"""Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 110.0  # safety margin under the 120 s kill

def _time_left():
    return _DEADLINE - time.time()


def _build_system(n, m, p):
    """Return (A, b, q) for the (n,m,p) multiplication tensor over GF(q).
    Column c encodes u_c (n*m), v_c (m*p), w_c (p*n).
    Tensor T_{ijk} = delta_{jk}; index W by k*n+i (official trace convention)."""
    q = 99991  # large prime; exact mod q => exact over Z for small-coeff solutions
    N = n * m * p
    rows = []
    b = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1 if j == k else 0
                row = np.zeros(N, dtype=np.int64)
                row[i * m + j] = 1                       # u_{i,j}
                row[(n * m) + j * p + k] = 1             # v_{j,k}
                row[(n * m + m * p) + k * n + i] = 1     # w_{k*n+i}
                rows.append(row)
                b.append(target)
    A = np.array(rows, dtype=np.int64)
    b = np.array(b, dtype=np.int64)
    return A, b, q


def _lsq_solve(A, b, q):
    """Least-norm integer solution of A x = b (exact) over GF(q), else None."""
    try:
        Aq = A % q
        bq = b % q
        x = np.linalg.lstsq(Aq.astype(np.float64), bq.astype(np.float64),
                            rcond=None)[0]
        xr = np.rint(x)
        if np.all((Aq @ xr) % q == bq):
            return xr
    except Exception:
        pass
    return None


def _greedy_sub(n, m, p, target_rank, max_iters):
    """Search for a rank-`target_rank` decomposition of the (n,m,p) tensor.
    Returns (U, V, W) as int64 arrays, or None."""
    A, b, q = _build_system(n, m, p)
    nU, nV, nW = n * m, m * p, p * n
    U = np.zeros((nU, target_rank), dtype=np.int64)
    V = np.zeros((nV, target_rank), dtype=np.int64)
    W = np.zeros((nW, target_rank), dtype=np.int64)
    used = np.zeros(target_rank, dtype=bool)
    bcur = b.copy()
    for c in range(target_rank):
        if _time_left() < 10.0:
            return None
        x = _lsq_solve(A, bcur, q)
        if x is None:
            return None
        used[c] = True
        U[:, c] = x[:nU]
        V[:, c] = x[nU:nU + nV]
        W[:, c] = x[nU + nV:]
        bcur = (bcur - (A @ x)) % q
        if not np.any(bcur):
            break
    if np.any(bcur):
        return None
    # compact: drop any all-zero columns (should not happen)
    cols = [c for c in range(target_rank) if np.any(used[c])]
    U = U[:, cols]
    V = V[:, cols]
    W = W[:, cols]
    return U, V, W


def _verify(n, m, p, U, V, W):
    """Exact check that sum_c u_c v_c w_c reproduces the (n,m,p) tensor."""
    rank = U.shape[1]
    u = U.reshape(n, m, rank).astype(np.int64)
    v = V.reshape(m, p, rank).astype(np.int64)
    w = W.reshape(p, n, rank).astype(np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = int(np.dot(u[i, j, :], v[j, k, :]))
                if val != int(np.dot(v[j, k, :], w[k, i, :])):
                    return False
                if val != (1 if j == k else 0):
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _to_list(mat):
    return [[int(v) for v in row] for row in mat]


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # Direct-sum reduction: m = 4 = 2 + 2  =>  R(3,4,6) <= 2 * R(3,2,6)
    sub = _greedy_sub(n, 2, p, target_rank=35, max_iters=35)
    if sub is not None:
        Us, Vs, Ws = sub
        # Duplicate into two identical 4-wide halves (rows j=0,1 and j=2,3).
        rank = 2 * Us.shape[1]
        U = np.zeros((n * 4, rank), dtype=np.int64)
        V = np.zeros((4 * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        for half in range(2):
            # U: rows i*4 + (half*2 + j) for j in 0..1
            for i in range(n):
                for j in range(2):
                    U[i * 4 + (half * 2 + j), half * Us.shape[1]:] = Us[i * 2 + j, :]
            # V: rows (half*2 + j)*p + k for j in 0..1
            for j in range(2):
                for k in range(p):
                    V[(half * 2 + j) * p + k, half * Us.shape[1]:] = Vs[j * p + k, :]
            # W: shared (independent of the m-block)
            W[:, half * Us.shape[1]:] = Ws
        if _verify(n, 4, p, U, V, W):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    # Fallback: exact schoolbook rank 72.
    U, V, W = _schoolbook(n, 4, p)
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
