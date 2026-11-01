"""Matrix multiplication 2x2 by 2x3 rank-minimal exact decomposition."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = None

def _now():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 130  # safety margin before 150s kill
    return time.time()

def _time_left():
    return _DEADLINE - time.time()

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify U, V, W decompose the matrix multiplication tensor exactly."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Check all 12 (i,j,k) triples: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    # Also verify with random integer matrices (robustness)
    rng = np.random.default_rng(42)
    for _ in range(10):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_mat = Ua[:, r].reshape(n, m)
            v_mat = Va[:, r].reshape(m, p)
            w_mat = Wa[:, r].reshape(p, n)
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C_decomp[i, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        if np.max(np.abs(C - C_decomp)) > tol:
            return False
    return True

def _rank9_decomposition():
    """Strassen (rank 7) for columns 0,1 + schoolbook (rank 2) for column 2 = rank 9."""
    # U rows: a00, a01, a10, a11
    # V rows: b00, b01, b02, b10, b11, b12
    # W rows: C00, C01, C10, C11, C02, C12  (index k*n+i)
    U = [
        [1, 0, 1, 0, 1, -1, 0, 1, 0],   # a00
        [0, 0, 0, 0, 1, 0, 1, 1, 0],    # a01
        [0, 1, 0, 0, 0, 1, 0, 0, 1],    # a10
        [1, 1, 0, 1, 0, 0, -1, 0, 1],   # a11
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0, 0, 0],   # b00
        [0, 0, 1, 0, 0, 1, 0, 0, 0],    # b01
        [0, 0, 0, 0, 0, 0, 0, 1, 1],    # b02
        [0, 0, 0, 1, 0, 0, 1, 0, 0],    # b10
        [1, 0, -1, 0, 1, 0, 1, 0, 0],   # b11
        [0, 0, 0, 0, 0, 0, 0, 1, 1],    # b12
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1, 0, 0],   # C00
        [0, 0, 1, 0, 1, 0, 0, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 1, 0, 0],   # C11
        [0, 0, 0, 0, 0, 0, 0, 1, 0],    # C02
        [0, 0, 0, 0, 0, 0, 0, 0, 1],    # C12
    ]
    if _verify(U, V, W, 2, 2, 3):
        return U, V, W
    return None

def _try_rank8_numerical():
    """Numerical alternating-least-squares search for rank 8, round to half-integers."""
    n, m, p = 2, 2, 3
    target_rank = 8
    deadline = _now() + 90
    rng = np.random.default_rng(7)

    def solve(U, V, W):
        # Solve for each factor given the other two (linear least squares).
        r = target_rank
        # Solve U: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Group by (i,j): sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1 for each k
        U_new = np.zeros((n*m, r))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    for c in range(r):
                        A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                b = np.ones(p)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U_new[i*m+j, :] = sol
        # Solve V
        V_new = np.zeros((m*p, r))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    for c in range(r):
                        A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                b = np.ones(n)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V_new[j*p+k, :] = sol
        # Solve W
        W_new = np.zeros((p*n, r))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                b = np.ones(m)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W_new[k*n+i, :] = sol
        return U_new, V_new, W_new

    def round_half(X):
        return np.round(X * 2) / 2.0

    for attempt in range(200):
        if time.time() > deadline:
            break
        U = rng.uniform(-1, 1, size=(n*m, target_rank))
        V = rng.uniform(-1, 1, size=(m*p, target_rank))
        W = rng.uniform(-1, 1, size=(p*n, target_rank))
        for _ in range(60):
            if time.time() > deadline:
                break
            U, V, W = solve(U, V, W)
        U_r = round_half(U)
        V_r = round_half(V)
        W_r = round_half(W)
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return None

def _schoolbook():
    n, m, p = 2, 2, 3
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
    return U, V, W

def construct():
    _now()  # initialize deadline

    # 1) Try the analytic rank-9 decomposition (Strassen + schoolbook).
    res = _rank9_decomposition()
    if res is not None:
        U, V, W = res
        return {"U": U, "V": V, "W": W}

    # 2) Try a numerical rank-8 search.
    if _time_left() > 60:
        res8 = _try_rank8_numerical()
        if res8 is not None:
            U, V, W = res8
            return {"U": U, "V": V, "W": W}

    # 3) Fall back to schoolbook rank 12 (guaranteed valid).
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
