import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _time_remaining():
    return _DEADLINE - time.time()

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition (always valid)."""
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_block_decomposition():
    """
    Correct rank-25 decomposition for 2x2 * 2x7:
    - Strassen rank-7 for columns 0,1
    - Strassen rank-7 for columns 2,3
    - Strassen rank-7 for columns 4,5
    - Schoolbook rank-4 for column 6
    """
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    def add_strassen(col_start, k0):
        # A = [[A00,A01],[A10,A11]], A index: 0=A00,1=A01,2=A10,3=A11
        # B cols k0,k0+1 ; B index: row0: 7*k0..7*k0+6, row1: 7+7*k0..
        # C index: 2*col + row  (W[k*n+i])
        # M1 = (A00+A11)*(B0,k0 + B1,k0+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k0, col_start+0] += 1
        V[7*1 + (k0+1), col_start+0] += 1
        W[2*k0 + 0, col_start+0] += 1
        W[2*(k0+1) + 1, col_start+0] += 1
        # M2 = (A10+A11)*B0,k0
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k0, col_start+1] += 1
        W[2*k0 + 1, col_start+1] += 1
        W[2*(k0+1) + 1, col_start+1] -= 1
        # M3 = A00*(B0,k0+1 - B1,k0+1)
        U[0, col_start+2] += 1
        V[7*0 + (k0+1), col_start+2] += 1
        V[7*1 + (k0+1), col_start+2] -= 1
        W[2*(k0+1) + 0, col_start+2] += 1
        W[2*(k0+1) + 1, col_start+2] += 1
        # M4 = A11*(B1,k0 - B0,k0)
        U[3, col_start+3] += 1
        V[7*1 + k0, col_start+3] += 1
        V[7*0 + k0, col_start+3] -= 1
        W[2*k0 + 0, col_start+3] += 1
        W[2*k0 + 1, col_start+3] += 1
        # M5 = (A00+A01)*B1,k0+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k0+1), col_start+4] += 1
        W[2*k0 + 0, col_start+4] -= 1
        W[2*(k0+1) + 0, col_start+4] += 1
        # M6 = (A10-A00)*(B0,k0 + B0,k0+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k0, col_start+5] += 1
        V[7*0 + (k0+1), col_start+5] += 1
        W[2*(k0+1) + 1, col_start+5] += 1
        # M7 = (A01-A11)*(B1,k0 + B1,k0+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k0, col_start+6] += 1
        V[7*1 + (k0+1), col_start+6] += 1
        W[2*k0 + 0, col_start+6] += 1

    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)

    # Schoolbook for column 6
    col_start = 21
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, num_trials=20):
    """Verify the decomposition exactly computes matrix multiplication."""
    n, m, p = 2, 2, 7
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    rng = np.random.default_rng(12345)
    for _ in range(num_trials):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(rank):
            SA = np.dot(U[:, r], A.flatten())
            SB = np.dot(V[:, r], B.flatten())
            prod = SA * SB
            for k in range(p):
                for i in range(n):
                    C_computed[i, k] += W[k*n + i, r] * prod
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _build_target_tensor():
    """Build the matrix-multiplication tensor as a 4 x (14*14) matrix.
    Entry (a, (b,c)) = 1 if C_{a/c} = sum_b A_{a/b} B_{b/c}.
    """
    n, m, p = 2, 2, 7
    T = np.zeros((n*m, m*p * p*n))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                a_idx = i*m + j
                bc_idx = (j*p + k) * (p*n) + (k*n + i)
                T[a_idx, bc_idx] = 1.0
    return T

def _matmul_residual(U, V, W):
    """Compute ||U diag(W) V^T - T||_F for a candidate decomposition."""
    n, m, p = 2, 2, 7
    T = _build_target_tensor()
    rank = U.shape[1]
    # M = sum_r U_r * W_r * V_r^T  ->  (n*m) x (m*p * p*n)
    M = np.zeros((n*m, m*p * p*n))
    for r in range(rank):
        Mr = np.outer(U[:, r], V[:, r]) * W[:, r]
        M += Mr
    return np.linalg.norm(M - T)

def _try_rank24():
    """
    Attempt a rank-24 decomposition via a smooth (least-squares) surrogate.
    Start from the rank-25 solution, drop one term, then use L-BFGS-B on the
    free variables (flattened U,V,W) to drive the residual toward zero.
    Returns a dict if a valid rank-24 decomposition is found, else None.
    """
    if _time_remaining() < 25.0:
        return None
    base = _strassen_block_decomposition()
    U0 = np.array(base['U'], dtype=float)
    V0 = np.array(base['V'], dtype=float)
    W0 = np.array(base['W'], dtype=float)
    n, m, p = 2, 2, 7
    T = _build_target_tensor()
    target_norm = np.linalg.norm(T)

    # Try dropping each of the 25 terms and polishing.
    for drop in range(25):
        if _time_remaining() < 10.0:
            break
        U = np.delete(U0, drop, axis=1)
        V = np.delete(V0, drop, axis=1)
        W = np.delete(W0, drop, axis=1)
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        def unpack(x):
            sz = U.size
            Uv = x[:sz].reshape(U.shape)
            Vv = x[sz:2*sz].reshape(V.shape)
            Wv = x[2*sz:3*sz].reshape(W.shape)
            return Uv, Vv, Wv

        def obj(x):
            Uv, Vv, Wv = unpack(x)
            M = np.zeros((n*m, m*p * p*n))
            for r in range(Uv.shape[1]):
                M += np.outer(Uv[:, r], Vv[:, r]) * Wv[:, r]
            return np.sum((M - T) ** 2)

        try:
            from scipy.optimize import minimize
            res = minimize(obj, x0, method='L-BFGS-B',
                           options={'maxiter': 400, 'maxls': 40,
                                    'gtol': 1e-6, 'maxfun': 20000})
        except Exception:
            continue
        Uv, Vv, Wv = unpack(res.x)
        # Check validity with a strict tolerance.
        if _verify_decomposition(Uv.tolist(), Vv.tolist(), Wv.tolist(), num_trials=10):
            return {"U": Uv.tolist(), "V": Vv.tolist(), "W": Wv.tolist()}
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x7.
    Returns a rank-24 decomposition if found within the time budget,
    otherwise a verified rank-25 decomposition, otherwise schoolbook rank-28.
    """
    # Always have a valid fallback ready.
    try:
        base = _strassen_block_decomposition()
        if _verify_decomposition(base['U'], base['V'], base['W'], num_trials=15):
            best = base
        else:
            best = _schoolbook_decomposition()
    except Exception:
        best = _schoolbook_decomposition()

    # Attempt to improve to rank 24 if we have budget.
    if _time_remaining() > 25.0:
        try:
            better = _try_rank24()
            if better is not None and _verify_decomposition(
                    better['U'], better['V'], better['W'], num_trials=15):
                best = better
        except Exception:
            pass

    return best

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print("rank =", U.shape[1])
    print("valid =", _verify_decomposition(res['U'], res['V'], res['W'], num_trials=20))
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        for idx in range(14):
            C_W[idx] += W[idx, r] * SA * SB
    C_mat = np.zeros((2, 7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k + i]
    print("match =", np.allclose(C_mat, C_expected))
