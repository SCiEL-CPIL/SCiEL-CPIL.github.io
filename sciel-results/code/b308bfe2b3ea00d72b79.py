"""Matrix multiplication 3,3,3 - rank-minimal exact tensor decomposition.

Uses a numerical solver to find a valid rank-23 decomposition,
falling back to rank-26 if the solver fails or times out.
"""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 110  # safety margin under 120s

    n, m, p = 3, 3, 3

    def verify(U_mat, V_mat, W_mat, trials=10):
        rng = np.random.default_rng(42)
        for _ in range(trials):
            A = rng.integers(-5, 5, size=(n, m))
            B = rng.integers(-5, 5, size=(m, p))
            C_expected = A @ B
            C_flat = np.zeros(p * n)
            for t in range(U_mat.shape[1]):
                a_part = U_mat[:, t] @ A.reshape(n * m)
                b_part = V_mat[:, t] @ B.reshape(m * p)
                C_flat += W_mat[:, t] * (a_part * b_part)
            C_recon = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    C_recon[i, k] = C_flat[k * n + i]
            if not np.allclose(C_recon, C_expected, atol=1e-9):
                return False
        return True

    def to_list(mat):
        rows = []
        for row in mat:
            r = []
            for v in row:
                # Round to nearest 0.5
                val = round(v * 2) / 2.0
                if abs(val - round(val)) < 1e-9:
                    r.append(int(round(val)))
                else:
                    r.append(val)
            rows.append(r)
        return rows

    # Try to find rank 23 decomposition
    if time.time() - start < DEADLINE - 10:
        U23, V23, W23 = _solve_rank(23, start, DEADLINE)
        if U23 is not None and verify(U23, V23, W23):
            return {"U": to_list(U23), "V": to_list(V23), "W": to_list(W23)}

    # Try rank 24
    if time.time() - start < DEADLINE - 5:
        U24, V24, W24 = _solve_rank(24, start, DEADLINE)
        if U24 is not None and verify(U24, V24, W24):
            return {"U": to_list(U24), "V": to_list(V24), "W": to_list(W24)}

    # Fallback to rank 26
    U26, V26, W26 = _rank26_fallback()
    if verify(U26, V26, W26):
        return {"U": to_list(U26), "V": to_list(V26), "W": to_list(W26)}

    # Last resort
    U27, V27, W27 = _rank27_schoolbook()
    return {"U": to_list(U27), "V": to_list(V27), "W": to_list(W27)}


def _solve_rank(rank, start_time, deadline):
    n, m, p = 3, 3, 3
    size_U = n * m
    size_V = m * p
    size_W = p * n
    total_vars = rank * (size_U + size_V + size_W)
    
    # Initialize with random values
    rng = np.random.default_rng(123)
    x0 = rng.normal(0, 1, total_vars)
    
    def residuals(x):
        if time.time() > deadline - 5:
            return None
        U = x[:rank*size_U].reshape(size_U, rank)
        V = x[rank*size_U:rank*(size_U+size_V)].reshape(size_V, rank)
        W = x[rank*(size_U+size_V):].reshape(size_W, rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        val = 0.0
                        for t in range(rank):
                            val += W[k*n+i, t] * U[i*m+j, t] * V[jp*p+k, t]
                        target = 1.0 if j == jp else 0.0
                        res.append(val - target)
        return np.array(res)
    
    # Check initial guess
    res0 = residuals(x0)
    if res0 is None:
        return None, None, None
    if np.linalg.norm(res0) < 1e-9:
        U = x0[:rank*size_U].reshape(size_U, rank)
        V = x0[rank*size_U:rank*(size_U+size_V)].reshape(size_V, rank)
        W = x0[rank*(size_U+size_V):].reshape(size_W, rank)
        return U, V, W

    try:
        sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
        if time.time() > deadline - 5:
            return None, None, None
        
        x_sol = sol.x
        U = x_sol[:rank*size_U].reshape(size_U, rank)
        V = x_sol[rank*size_U:rank*(size_U+size_V)].reshape(size_V, rank)
        W = x_sol[rank*(size_U+size_V):].reshape(size_W, rank)
        
        # Round to nearest 0.5 and verify
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        
        # Check if rounded solution is valid
        if _verify_exact(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        
        # Check if unrounded solution is valid (within tolerance)
        if _verify_exact(U, V, W, tol=1e-9):
            return U, V, W
            
    except Exception:
        pass
        
    return None, None, None

def _verify_exact(U, V, W, tol=1e-9):
    n, m, p = 3, 3, 3
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(U.shape[1]):
                        val += W[k*n+i, t] * U[i*m+j, t] * V[jp*p+k, t]
                    target = 1.0 if j == jp else 0.0
                    if abs(val - target) > tol:
                        return False
    return True

def _rank26_fallback():
    """The verified rank-26 decomposition from the seed program."""
    n, m, p = 3, 3, 3
    rank = 26
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx, col] += coeff
        V[v_idx, col] = 1
        W[w_idx, col] = 1
        col += 1

    # Strassen 2x2
    U[0, col] = 1; U[4, col] = 1
    V[0, col] = 1; V[4, col] = 1
    W[0, col] = 1; W[4, col] = 1
    col += 1
    U[3, col] = 1; U[4, col] = 1
    V[0, col] = 1
    W[1, col] = 1; W[4, col] = -1
    col += 1
    U[0, col] = 1
    V[1, col] = 1; V[4, col] = -1
    W[3, col] = 1; W[4, col] = 1
    col += 1
    U[4, col] = 1
    V[3, col] = 1; V[0, col] = -1
    W[0, col] = 1; W[1, col] = 1
    col += 1
    U[0, col] = 1; U[1, col] = 1
    V[4, col] = 1
    W[0, col] = -1; W[3, col] = 1
    col += 1
    U[3, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[4, col] = 1
    col += 1
    U[1, col] = 1; U[4, col] = -1
    V[3, col] = 1; V[4, col] = 1
    W[0, col] = 1
    col += 1

    # Cross terms
    add_term(2, 6, 0)
    add_term(2, 7, 3)
    add_term(5, 6, 1)
    add_term(5, 7, 4)
    add_term(0, 2, 6)
    add_term(1, 5, 6)
    add_term(3, 2, 7)
    add_term(4, 5, 7)
    add_term(2, 8, 6)
    add_term(5, 8, 7)
    add_term(6, 0, 2)
    add_term(7, 3, 2)
    add_term(6, 1, 5)
    add_term(7, 4, 5)
    add_term(8, 6, 2)
    add_term(8, 7, 5)
    add_term(6, 2, 8)
    add_term(7, 5, 8)
    add_term(8, 8, 8)

    return U, V, W


def _rank27_schoolbook():
    """Standard schoolbook rank-27 decomposition."""
    n, m, p = 3, 3, 3
    rank = 27
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t = i * m * p + j * p + k
                U[i * m + j, t] = 1
                V[j * p + k, t] = 1
                W[k * n + i, t] = 1
    return U, V, W
# EVOLVE-BLOCK-END
