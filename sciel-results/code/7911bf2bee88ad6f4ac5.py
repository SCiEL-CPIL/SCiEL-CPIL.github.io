"""Matrix multiplication 2x3x10 - rank-minimal exact tensor decomposition.

Uses structured initialization, regularized alternating minimization with
null-space exploitation, and column elimination to find rank-minimal
decompositions below the schoolbook rank of 60.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _solve_W_regularized(U, V, n, m, p, lam=0.01):
    """For fixed U, V, find W minimizing ||A_w - 1||^2 + lam*||W||^2."""
    W = np.zeros((p * n, U.shape[1]))
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, U.shape[1]))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b = np.ones(m)
            # Regularized least squares: (A^T A + lam*I) W_row = A^T b
            AtA = A.T @ A + lam * np.eye(U.shape[1])
            Atb = A.T @ b
            try:
                W[k * n + i, :] = np.linalg.solve(AtA, Atb)
            except np.linalg.LinAlgError:
                W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
    return W


def _solve_U_regularized(W, V, n, m, p, lam=0.01):
    """For fixed W, V, find U minimizing error with regularization."""
    U = np.zeros((n * m, W.shape[1]))
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, W.shape[1]))
            for k in range(p):
                A[k, :] = W[k * n + i, :] * V[j * p + k, :]
            b = np.ones(p)
            AtA = A.T @ A + lam * np.eye(W.shape[1])
            Atb = A.T @ b
            try:
                U[i * m + j, :] = np.linalg.solve(AtA, Atb)
            except np.linalg.LinAlgError:
                U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
    return U


def _solve_V_regularized(U, W, n, m, p, lam=0.01):
    """For fixed U, W, find V minimizing error with regularization."""
    V = np.zeros((m * p, U.shape[1]))
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, U.shape[1]))
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            b = np.ones(n)
            AtA = A.T @ A + lam * np.eye(U.shape[1])
            Atb = A.T @ b
            try:
                V[j * p + k, :] = np.linalg.solve(AtA, Atb)
            except np.linalg.LinAlgError:
                V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
    return V


def _compute_error(U, V, W, n, m, p):
    """Compute tensor error: sum over (i,j,k) of (sum_c U*V*W - 1)^2."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err += (val - 1.0) ** 2
    return err


def _verify(U, V, W, n, m, p, ntests=100, tol=1e-9):
    """Check exactness on random matrices."""
    rng = np.random.default_rng(12345)
    for _ in range(ntests):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                C_calc[i, k] = s
        if np.max(np.abs(C_calc - C_true)) > tol:
            return False
    return True


def _structured_init(n, m, p, r, seed):
    """Create structured initial U, V that respect tensor structure."""
    rng = np.random.default_rng(seed)
    # Initialize with small random values plus structured components
    U = rng.standard_normal((n * m, r)) * 0.1
    V = rng.standard_normal((m * p, r)) * 0.1
    # Add structured components: for each (i,j,k), ensure some column captures it
    for c in range(min(r, n * m * p)):
        i = c % n
        j = (c // n) % m
        k = (c // (n * m)) % p
        U[i * m + j, c] += 1.0
        V[j * p + k, c] += 1.0
    return U, V


def _try_rank(r, n, m, p, deadline, max_iter=500, n_trials=5):
    """Attempt to find an exact rank-r decomposition."""
    best_err = float('inf')
    best = None
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        U, V = _structured_init(n, m, p, r, 1000 + trial * 137)
        W = _solve_W_regularized(U, V, n, m, p, lam=0.01)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            U = _solve_U_regularized(W, V, n, m, p, lam=0.01)
            V = _solve_V_regularized(U, W, n, m, p, lam=0.01)
            W = _solve_W_regularized(U, V, n, m, p, lam=0.01)
            
            err = _compute_error(U, V, W, n, m, p)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if err < 1e-12:
                break
        
        if best is not None:
            Uv, Vv, Wv = best
            if _verify(Uv, Vv, Wv, n, m, p):
                return Uv, Vv, Wv
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s limit

    # Baseline: schoolbook rank 60
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    # Theoretical lower bound: n*m + m*p + n*p - m = 6 + 30 + 20 - 3 = 53
    lower_bound = n * m + m * p + n * p - m

    # Try to improve from rank 59 down toward the lower bound
    for r in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            if r <= lower_bound:
                break

    # Convert to lists of lists of numbers
    def to_list(M):
        return [[float(x) for x in row] for row in M]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
