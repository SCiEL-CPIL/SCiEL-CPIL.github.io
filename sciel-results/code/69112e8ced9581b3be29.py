"""Find a rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the trivial (schoolbook) decomposition with rank = n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-7):
    """Check that the decomposition is exact: T[i,j,k] = 1 for all (i,j,k)."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[d, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_rank(rank, n, m, p, deadline):
    """
    Try to find a valid decomposition of the given rank using iterative
    alternating least squares with random restarts.
    
    The key idea: fix two of the three matrices and solve for the third via
    a linear system. Iterate until convergence or deadline.
    """
    nm, mp, pn = n * m, m * p, p * n
    total_vars = nm + mp + pn  # 47

    # Precompute index mappings for efficiency
    # For each (i,j,k): a=i*m+j, b=j*p+k, d=k*n+i
    idx_a = []
    idx_b = []
    idx_d = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a.append(i * m + j)
                idx_b.append(j * p + k)
                idx_d.append(k * n + i)
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_d = np.array(idx_d)
    n_constraints = len(idx_a)  # 60

    best_residual = float('inf')
    best_solution = None

    for restart in range(8):
        if time.time() > deadline - 5:
            break

        np.random.seed(restart * 7919 + rank * 31)
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5

        for iteration in range(200):
            if time.time() > deadline - 5:
                break

            # Solve for U given V, W:
            # For each (i,j,k): sum_c U[a,c] * (V[b,c] * W[d,c]) = 1
            # This is a linear system in U's entries.
            # For each a (row of U), the equations involving U[a,:] are those
            # where idx_a == a. For each such equation, the coefficient of U[a,c]
            # is V[idx_b, c] * W[idx_d, c].

            # Build the linear system: A_U * U_flat = 1
            # U_flat has nm*rank entries.
            # For equation e (corresponding to (i,j,k)):
            #   sum_c U[idx_a[e], c] * (V[idx_b[e], c] * W[idx_d[e], c]) = 1

            A_U = np.zeros((n_constraints, nm * rank))
            for e in range(n_constraints):
                a = idx_a[e]
                b = idx_b[e]
                d = idx_d[e]
                coeff = V[b, :] * W[d, :]  # shape (rank,)
                for c in range(rank):
                    A_U[e, a * rank + c] = coeff[c]

            try:
                U_new, residuals, rank_u, sv = np.linalg.lstsq(A_U, np.ones(n_constraints), rcond=None)
                U_new = U_new.reshape(nm, rank)
            except np.linalg.LinAlgError:
                break

            # Solve for V given U, W
            A_V = np.zeros((n_constraints, mp * rank))
            for e in range(n_constraints):
                a = idx_a[e]
                b = idx_b[e]
                d = idx_d[e]
                coeff = U[a, :] * W[d, :]
                for c in range(rank):
                    A_V[e, b * rank + c] = coeff[c]

            try:
                V_new, _, _, _ = np.linalg.lstsq(A_V, np.ones(n_constraints), rcond=None)
                V_new = V_new.reshape(mp, rank)
            except np.linalg.LinAlgError:
                break

            # Solve for W given U, V
            A_W = np.zeros((n_constraints, pn * rank))
            for e in range(n_constraints):
                a = idx_a[e]
                b = idx_b[e]
                d = idx_d[e]
                coeff = U[a, :] * V[b, :]
                for c in range(rank):
                    A_W[e, d * rank + c] = coeff[c]

            try:
                W_new, _, _, _ = np.linalg.lstsq(A_W, np.ones(n_constraints), rcond=None)
                W_new = W_new.reshape(pn, rank)
            except np.linalg.LinAlgError:
                break

            U, V, W = U_new, V_new, W_new

            # Check residual
            max_res = 0.0
            for e in range(n_constraints):
                a, b, d = idx_a[e], idx_b[e], idx_d[e]
                val = np.dot(U[a, :], V[b, :] * W[d, :])
                max_res = max(max_res, abs(val - 1.0))

            if max_res < best_residual:
                best_residual = max_res
                best_solution = (U.copy(), V.copy(), W.copy())

            if max_res < 1e-7:
                return U.tolist(), V.tolist(), W.tolist()

    # Return best found
    if best_solution is not None:
        U, V, W = best_solution
        if best_residual < 1e-6:
            return U.tolist(), V.tolist(), W.tolist()

    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Lower bound: max(n*p, m*p, n*m) = max(15, 20, 12) = 20
    # But the true rank is likely higher. Let's try from 20 upward.
    # For efficiency, try a few candidate ranks.
    
    # The schoolbook rank is 60. We want to beat that.
    # Known results: for small matrices, rank is often less than schoolbook.
    # Let's try ranks from 20 to 59.
    
    for rank in range(20, 60):
        if time.time() > deadline - 10:
            break

        result = _try_rank(rank, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            # Final verification
            if _verify(U, V, W, tol=1e-6):
                return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
