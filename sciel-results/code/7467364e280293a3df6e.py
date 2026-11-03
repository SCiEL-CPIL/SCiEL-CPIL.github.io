"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Strategy:
1. Construct the known rank-55 decomposition using Strassen's algorithm
   on the 2x2 sub-blocks (j=1,2) and naive multiplication for j=0.
2. Exploit the bilinear structure: fixing two factors makes the third
   solvable via a linear system. Use alternating linear solves (ALS)
   to polish and to test rank-54 feasibility.
3. For rank 54, drop one term, re-solve W (linear), then polish U,V,W
   with alternating linear solves. If the residual goes to zero, return.
"""

import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]


def _nonzero_indices(n, m, p):
    """List of (a,b,c) indices where the mult tensor equals 1."""
    idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx.append((i * m + j, j * p + k, k * n + i))
    return idx


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition. U:(n*m,R) V:(m*p,R) W:(p*n,R)."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    R = U.shape[1]
    # Build tensor efficiently
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Vectorize: for each r, outer(U[:,r], V[:,r], W[:,r])
    # Use einsum
    T = np.einsum('ar,br,cr->abc', U, V, W)
    # Check nonzero entries
    for (a, b, c) in _nonzero_indices(n, m, p):
        if abs(T[a, b, c] - 1.0) > tol:
            return False
    # Check that all other entries are zero (full check for small tensor)
    nz_set = set(_nonzero_indices(n, m, p))
    # Zero out known entries, check rest
    T2 = T.copy()
    for (a, b, c) in nz_set:
        T2[a, b, c] = 0.0
    if np.max(np.abs(T2)) > tol:
        return False
    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _round_half(U, V, W):
    """Round all entries to nearest half-integer."""
    def r(M):
        return np.round(np.asarray(M, dtype=np.float64) * 2.0) / 2.0
    return r(U), r(V), r(W)


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block (j=1,2).
    """
    rank = 55
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0

    # Part A: j=0 (naive)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    # Part B: j=1,2 (Strassen on 2x2 blocks over k pairs)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1 = (A11+A22)(B11+B22)
        U[u_11, col] += 1; U[u_22, col] += 1
        V[v_11, col] += 1; V[v_22, col] += 1
        W[w_00, col] += 1; W[w_11, col] += 1
        col += 1
        # M2 = A21(B11+B12)
        U[u_21, col] += 1; U[u_22, col] += 1
        V[v_11, col] += 1
        W[w_10, col] += 1; W[w_11, col] += -1
        col += 1
        # M3 = A11(B22-B21)
        U[u_11, col] += 1
        V[v_12, col] += 1; V[v_22, col] += -1
        W[w_01, col] += 1; W[w_11, col] += 1
        col += 1
        # M4 = A22(B11-B12)
        U[u_22, col] += 1
        V[v_21, col] += 1; V[v_11, col] += -1
        W[w_00, col] += 1; W[w_10, col] += 1
        col += 1
        # M5 = (A11+A12)B22
        U[u_11, col] += 1; U[u_12, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += -1; W[w_01, col] += 1
        col += 1
        # M6 = (A21-A11)B11
        U[u_21, col] += 1; U[u_11, col] += -1
        V[v_11, col] += 1; V[v_12, col] += 1
        W[w_11, col] += 1
        col += 1
        # M7 = (A12-A22)(B21+B22)
        U[u_12, col] += 1; U[u_22, col] += -1
        V[v_21, col] += 1; V[v_22, col] += 1
        W[w_00, col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _residual(U, V, W, n, m, p):
    """Squared Frobenius norm of the error tensor."""
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T2 = T.copy()
    for (a, b, c) in _nonzero_indices(n, m, p):
        T2[a, b, c] -= 1.0
    return float(np.sum(T2 ** 2))


def _solve_W_given_UV(U, V, n, m, p):
    """
    Fix U and V. Solve for W via linear systems.
    For each (i,k): (U[i,:,r] * W[k,i,r]) dot V[:,k,r] = I_m  (over r)
    i.e., sum_r U[i,j,r]*W[k,i,r]*V[j',k,r] = delta_{j,j'}
    This is linear in W[k,i,r]. Build a linear system: for each (i,k),
    unknowns are W[k,i,0..R-1]. Equations: for each (j,j') in m*m,
    sum_r U[i,j,r]*V[j',k,r]*W[k,i,r] = delta_{j,j'}.
    """
    R = U.shape[1]
    W = np.zeros((p * n, R), dtype=np.float64)
    # Precompute: for each (i,k), the coefficient matrix A_{(j,j'),r} = U[i,j,r]*V[j',k,r]
    for i in range(n):
        for k in range(p):
            # A is (m*m, R)
            A = np.zeros((m * m, R), dtype=np.float64)
            for j in range(m):
                for jp in range(m):
                    A[j * m + jp, :] = U[i * m + j, :] * V[jp * p + k, :]
            # RHS: delta_{j,jp}
            rhs = np.zeros(m * m, dtype=np.float64)
            for j in range(m):
                rhs[j * m + j] = 1.0
            # Solve A w = rhs for w (length R). Use least squares.
            try:
                w_sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
            except np.linalg.LinAlgError:
                w_sol = np.zeros(R)
            W[k * n + i, :] = w_sol
    return W


def _solve_U_given_VW(V, W, n, m, p):
    """
    Fix V and W. Solve for U.
    For each (i,k): sum_r U[i,j,r]*V[j',k,r]*W[k,i,r] = delta_{j,j'}
    Linear in U[i,j,r]. Unknowns: U[i,:,0..R-1] (m*R unknowns).
    Equations: for each (j,jp), sum_r U[i,j,r]*V[jp,k,r]*W[k,i,r] = delta.
    But U[i,j,r] appears in equation (j,jp) for all jp.
    Actually for fixed i, the unknowns are U[i, j, r] for j in [0,m), r in [0,R).
    Equation (j, jp): sum_r U[i,j,r] * V[jp,k,r] * W[k,i,r] = delta_{j,jp}
    So for fixed i, we have m*m equations and m*R unknowns.
    Build system: unknown vector is U[i, j, r] flattened (m*R).
    Equation (j,jp): sum_r U[i,j,r]*V[jp,k,r]*W[k,i,r] = delta_{j,jp}
    """
    R = V.shape[1]
    U = np.zeros((n * m, R), dtype=np.float64)
    for i in range(n):
        for k in range(p):
            # Unknowns: U[i, j, r] for j in 0..m-1, r in 0..R-1 -> m*R unknowns
            # Equations: (j, jp) for j,jp in 0..m-1 -> m*m equations
            # Equation (j,jp): sum_r U[i,j,r] * V[jp,k,r] * W[k,i,r] = delta_{j,jp}
            # Coefficient of U[i,j,r] in equation (j,jp) is V[jp,k,r]*W[k,i,r] if row=j, else 0.
            # So the system is block-diagonal in j! For each j, unknowns U[i,j,0..R-1],
            # equations are (j, jp) for jp in 0..m-1 -> m equations, R unknowns.
            for j in range(m):
                # m equations, R unknowns
                A = np.zeros((m, R), dtype=np.float64)
                for jp in range(m):
                    A[jp, :] = V[jp * p + k, :] * W[k * n + i, :]
                rhs = np.zeros(m, dtype=np.float64)
                rhs[j] = 1.0
                try:
                    w_sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    w_sol = np.zeros(R)
                U[i * m + j, :] = w_sol
    return U


def _solve_V_given_UW(U, W, n, m, p):
    """
    Fix U and W. Solve for V.
    For each (i,k): sum_r U[i,j,r]*V[jp,k,r]*W[k,i,r] = delta_{j,jp}
    Linear in V[jp,k,r]. For fixed k, unknowns V[jp,k,r] for jp in 0..m-1, r in 0..R.
    Equation (j,jp): sum_r U[i,j,r]*V[jp,k,r]*W[k,i,r] = delta_{j,jp}
    For fixed k and jp, unknowns V[jp,k,0..R-1], equations over i and j:
    (i,j,jp): sum_r U[i,j,r]*V[jp,k,r]*W[k,i,r] = delta_{j,jp}
    So for fixed (k, jp), we have n*m equations (over i,j) and R unknowns.
    """
    R = U.shape[1]
    V = np.zeros((m * p, R), dtype=np.float64)
    for k in range(p):
        for jp in range(m):
            # Unknowns: V[jp,k,0..R-1] (R unknowns)
            # Equations: (i, j) for i in 0..n-1, j in 0..m-1 -> n*m equations
            A = np.zeros((n * m, R), dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    A[i * m + j, :] = U[i * m + j, :] * W[k * n + i, :]
            rhs = np.zeros(n * m, dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    if j == jp:
                        rhs[i * m + j] = 1.0
            try:
                w_sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
            except np.linalg.LinAlgError:
                w_sol = np.zeros(R)
            V[jp * p + k, :] = w_sol
    return V


def _als_polish(U, V, W, n, m, p, iters=50, tol=1e-12, deadline=None):
    """
    Alternating linear solves: fix two, solve third. Repeat.
    """
    R = U.shape[1]
    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        # Solve W given U,V
        W_new = _solve_W_given_UV(U, V, n, m, p)
        # Solve U given V,W
        U_new = _solve_U_given_VW(V, W_new, n, m, p)
        # Solve V given U,W
        V_new = _solve_V_given_UW(U_new, W_new, n, m, p)

        # Check convergence
        diff = (np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) +
                np.max(np.abs(W_new - W)))
        U, V, W = U_new, V_new, W_new
        if diff < tol:
            break
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s kill

    # Start with rank-55
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    # Verify
    if not (_verify(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Polish the rank-55 solution with ALS to ensure it's exact
    U_best, V_best, W_best = _als_polish(U_best, V_best, W_best, n, m, p,
                                          iters=20, deadline=deadline - 20)
    # Round to half-integers
    U_best, V_best, W_best = _round_half(U_best, V_best, W_best)
    if _verify(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best):
        best_rank = 55
    else:
        # Fallback to original
        U_best, V_best, W_best = _build_strassen55(n, m, p)
        best_rank = 55

    # Try to find rank-54 by dropping one term and polishing
    for r_remove in range(55):
        if time.time() > deadline - 15:
            break

        U_init = np.delete(U_best, r_remove, axis=1)
        V_init = np.delete(V_best, r_remove, axis=1)
        W_init = np.delete(W_best, r_remove, axis=1)

        # Solve W given U,V (linear)
        W_sol = _solve_W_given_UV(U_init, V_init, n, m, p)

        # Polish with ALS
        U_sol, V_sol, W_sol = _als_polish(U_init, V_init, W_sol, n, m, p,
                                           iters=30, deadline=deadline - 10)

        # Check residual
        res = _residual(U_sol, V_sol, W_sol, n, m, p)
        if res < 1e-10:
            # Found a valid rank-54 solution
            U_sol, V_sol, W_sol = _round_half(U_sol, V_sol, W_sol)
            if _verify(U_sol, V_sol, W_sol, n, m, p) and _is_valid_entries(U_sol, V_sol, W_sol):
                return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
            # Even if not half-integer, check exact
            if _verify(U_sol, V_sol, W_sol, n, m, p):
                # Try rounding
                U_r, V_r, W_r = _round_half(U_sol, V_sol, W_sol)
                if _verify(U_r, V_r, W_r, n, m, p):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Return the best known solution (rank 55)
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
