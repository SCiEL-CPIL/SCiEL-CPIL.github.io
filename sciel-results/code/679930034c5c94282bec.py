"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Strategy:
  1. Build a known rank-55 decomposition (Strassen on 2x2 blocks + naive 2x1*1x10).
  2. Formulate the problem as finding a sparse support pattern (binary variables)
     that can be lifted to exact rational coefficients.
  3. Use a structured search over support patterns derived from the rank-55
     decomposition, looking for rank 54 or lower.
  4. For each candidate support, solve the linear system for coefficients.
  5. Verify exactness and half-integer constraint.
"""

import numpy as np
import random
import time
from itertools import combinations

DIMENSIONS = [2, 3, 10]


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Fast verification using matrix multiplication.
    For each (i, k), check that sum_r U[i,j,r] * W[k,i,r] * V[j,k,r] == delta_jj'.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    for i in range(n):
        for k in range(p):
            # M_ik[j1, j2] = sum_r U[i, j1, r] * W[k, i, r] * V[j2, k, r]
            Uw = U_mat[i] * W_mat[k, i]  # shape (m, rank)
            M_ik = Uw @ V_mat[:, k].T  # shape (m, m)
            if not np.allclose(M_ik, np.eye(m), atol=tol):
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


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
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

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
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


def _solve_for_coefficients(U_support, V_support, W_support, n, m, p):
    """
    Given support patterns (binary matrices), solve for coefficients such that
    the decomposition is exact. Returns None if no solution exists or if the
    solution doesn't have half-integer entries.
    
    U_support, V_support, W_support are (rows, rank) binary matrices.
    We need to find diagonal scaling or more generally, we need to find
    U, V, W with the same support such that the tensor identity holds.
    
    Actually, the problem is: given that the support is fixed, find values
    on the non-zero entries. This is a system of quadratic equations in general,
    but if we fix the values in U and V, then W is determined linearly (or vice versa).
    
    Let's use the approach: fix U and V to have values in {1, -1} on their support
    (or 0.5, -0.5), then solve for W linearly. But this is restrictive.
    
    Better approach: The tensor identity is:
    sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    For a fixed support pattern, this is a system of equations in the unknown
    coefficients. Let x_r be the coefficient for term r (if we assume each term
    is a product of three vectors with a single scaling factor... no, that's not right).
    
    Actually, each term r has three vectors U[:,r], V[:,r], W[:,r]. The entries
    of these vectors are the unknowns. But the support fixes which entries are
    non-zero. So we have a system of polynomial equations.
    
    Simplification: If we assume that for each term r, the non-zero entries in
    U[:,r], V[:,r], W[:,r] are all ±1 or ±0.5, then we can search over signs.
    But that's exponential.
    
    Alternative: Use the fact that the rank-55 decomposition is known. We can
    try to merge two terms by finding a linear combination that preserves the
    tensor identity. This is what _try_merge does.
    
    Let's implement a direct approach: for a candidate support pattern, set up
    the linear system for W given U and V. But U and V also have unknowns.
    
    Most practical approach: Start from the rank-55 decomposition. Try to drop
    one term and re-solve for the coefficients of the remaining 54 terms using
    least squares, then check if the solution is exact and has valid entries.
    """
    pass


def _try_drop_term(U55, V55, W55, n, m, p, drop_idx, deadline):
    """
    Try to drop term drop_idx and re-solve for coefficients of remaining terms.
    """
    rank55 = 55
    keep_idx = [i for i in range(rank55) if i != drop_idx]
    U_keep = U55[:, keep_idx]
    V_keep = V55[:, keep_idx]
    W_keep = W55[:, keep_idx]
    
    # We want to find coefficients c_r such that:
    # sum_r c_r * (U_keep[:,r] ⊗ V_keep[:,r] ⊗ W_keep[:,r]) = T_mult
    # where T_mult is the matrix multiplication tensor.
    
    # Set up linear system: A * c = b
    # A is (n*m*p, 54) where A[(i,j,k), r] = U_keep[i*m+j, r] * V_keep[j*p+k, r] * W_keep[k*n+i, r]
    # b[(i,j,k)] = 1 if (i,j,k) is in the support of T_mult, else 0
    
    # The support of T_mult is: (i,j,k) where i in {0,1}, j in {0,1,2}, k in {0,...,9}
    # But actually, the tensor T_mult has T_mult[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # and 0 otherwise.
    
    # We only need to satisfy the equations for (i,j,k) in the support.
    # There are 2*3*10 = 60 such equations.
    
    # But we also need the other entries to be 0. However, if the support of the
    # candidate decomposition is a subset of the support of T_mult, then the
    # other entries are automatically 0. But that's not necessarily true.
    
    # Let's just solve for the 60 support equations and then verify the full tensor.
    
    eqs = []
    b_vec = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                row = []
                for r in range(54):
                    row.append(U_keep[a, r] * V_keep[b, r] * W_keep[c, r])
                eqs.append(row)
                b_vec.append(1.0)
    
    A = np.array(eqs, dtype=np.float64)
    b = np.array(b_vec, dtype=np.float64)
    
    try:
        coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    
    if len(residuals) > 0 and residuals[0] > 1e-8:
        return None
    
    # Check if coefficients are half-integers
    valid = True
    for cv in coeffs:
        if abs(cv * 2 - round(cv * 2)) > 1e-6:
            valid = False
            break
    if not valid:
        return None
    
    # Build new U, V, W
    U_new = U_keep * coeffs
    V_new = V_keep * coeffs
    W_new = W_keep * coeffs
    
    # Verify
    if _verify_fast(U_new, V_new, W_new, n, m, p):
        if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None


def _try_merge_terms(U55, V55, W55, n, m, p, deadline):
    """
    Try to merge two terms into one by finding a linear combination.
    """
    rank55 = 55
    
    for i in range(rank55):
        for j in range(i + 1, rank55):
            if time.time() > deadline:
                return None
            
            # Try to drop term i and see if we can express it as a combination of others
            # Or try to find a new term that replaces both i and j.
            
            # Simpler: try to drop term i and re-solve
            result = _try_drop_term(U55, V55, W55, n, m, p, i, deadline)
            if result is not None:
                return result
    
    return None


def _random_support_search(n, m, p, rank, deadline, U55, V55, W55):
    """
    Random search for a support pattern of the given rank.
    """
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)
    W55 = np.array(W55, dtype=np.float64)
    
    for trial in range(1000):
        if time.time() > deadline:
            break
        
        # Pick a random subset of terms from the rank-55 decomposition
        idx = random.sample(range(55), rank)
        U_trial = U55[:, idx].copy()
        V_trial = V55[:, idx].copy()
        W_trial = W55[:, idx].copy()
        
        if _verify_fast(U_trial, V_trial, W_trial, n, m, p):
            if _is_valid_entries(U_trial.tolist(), V_trial.tolist(), W_trial.tolist()):
                return U_trial.tolist(), V_trial.tolist(), W_trial.tolist()
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    if not (_verify_fast(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    U55 = np.array(U_best, dtype=np.float64)
    V55 = np.array(V_best, dtype=np.float64)
    W55 = np.array(W_best, dtype=np.float64)

    # Try to find rank 54 by dropping one term
    if time.time() < deadline - 20:
        result = _try_merge_terms(U55, V55, W55, n, m, p, deadline - 10)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 54
            return {"U": U_best, "V": V_best, "W": W_best}

    # Try random support search for rank 54
    if time.time() < deadline - 15:
        result = _random_support_search(n, m, p, 54, deadline - 10, U55, V55, W55)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 54
            return {"U": U_best, "V": V_best, "W": W_best}

    # Try rank 53
    if time.time() < deadline - 10:
        result = _random_support_search(n, m, p, 53, deadline - 5, U55, V55, W55)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 53
            return {"U": U_best, "V": V_best, "W": W_best}

    return {"U": U_best, "V": V_best, "W": W_best}
