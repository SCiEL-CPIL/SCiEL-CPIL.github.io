"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Design:
  - Start with a rank-55 Strassen-based decomposition (2x1 naive + 5x(2x2) Strassen).
  - Try to reduce rank by:
    1. Direct structural merges (drop one term, solve for it in the span of the rest).
    2. Local ALS polish around the known solution for ranks 54..50.
    3. Random linear combinations / subspace sampling.
  - Fast verification via per-(i,k) matrix identity check.
  - All entries must be integers or half-integers.
"""

import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]
NM, MP, PN = 6, 30, 20  # n*m, m*p, p*n


def _verify_fast(U, V, W, n=2, m=3, p=10, tol=1e-9):
    """
    Fast verification: for each (i,k), check that
    sum_r U[i,:,r] * W[k,i,r] * V[:,k,r] == I_m.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]

    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)

    for i in range(n):
        for k in range(p):
            # M_ik[j1,j2] = sum_r U[i,j1,r] * W[k,i,r] * V[j2,k,r]
            Uw = U_mat[i] * W_mat[k, i]          # (m, rank)
            M_ik = Uw @ V_mat[:, k].T            # (m, m)
            if not np.allclose(M_ik, np.eye(m), atol=tol):
                return False
    return True


def _verify_full(U, V, W, n=2, m=3, p=10, tol=1e-9):
    """Full verification of the tensor decomposition."""
    rank = len(U[0])
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    # Check all required entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.einsum('r,r,r->', U[a, :], V[b, :], W[c, :])
                if abs(val - 1.0) > tol:
                    return False

    # Check a sample of zero entries
    rng = np.random.RandomState(42)
    for _ in range(2000):
        a = rng.randint(0, n * m - 1)
        b = rng.randint(0, m * p - 1)
        c = rng.randint(0, p * n - 1)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if not (i == i2 and j == j2 and k == k2):
            val = np.einsum('r,r,r->', U[a, :], V[b, :], W[c, :])
            if abs(val) > tol:
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


def _build_strassen55(n=2, m=3, p=10):
    """
    Build the rank-55 decomposition:
      Part A: j=0, all (i,k): 2*10 = 20 terms
      Part B: j in {1,2}, 5 blocks of Strassen (7 each) = 35 terms
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Part A: j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part B: Strassen on 2x2 blocks
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u11 = 0 * m + 1; u12 = 0 * m + 2
        u21 = 1 * m + 1; u22 = 1 * m + 2

        v11 = 1 * p + k0; v12 = 1 * p + k1
        v21 = 2 * p + k0; v22 = 2 * p + k1

        w00 = k0 * n + 0; w01 = k1 * n + 0
        w10 = k0 * n + 1; w11 = k1 * n + 1

        # M1 = (a11+a22)(b11+b22) -> c00 += M1, c11 += M1
        U[u11][col] += 1; U[u22][col] += 1
        V[v11][col] += 1; V[v22][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1

        # M2 = (a21+a22)b11 -> c10 += M2, c11 -= M2
        U[u21][col] += 1; U[u22][col] += 1
        V[v11][col] += 1
        W[w10][col] += 1; W[w11][col] += -1
        col += 1

        # M3 = a11(b12-b22) -> c01 += M3, c11 += M3
        U[u11][col] += 1
        V[v12][col] += 1; V[v22][col] += -1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1

        # M4 = a22(b21-b11) -> c00 += M4, c10 += M4
        U[u22][col] += 1
        V[v21][col] += 1; V[v11][col] += -1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1

        # M5 = (a11+a12)b22 -> c00 -= M5, c01 += M5
        U[u11][col] += 1; U[u12][col] += 1
        V[v22][col] += 1
        W[w00][col] += -1; W[w01][col] += 1
        col += 1

        # M6 = (a21-a11)(b11+b12) -> c11 += M6
        U[u21][col] += 1; U[u11][col] += -1
        V[v11][col] += 1; V[v12][col] += 1
        W[w11][col] += 1
        col += 1

        # M7 = (a12-a22)(b21+b22) -> c00 += M7
        U[u12][col] += 1; U[u22][col] += -1
        V[v21][col] += 1; V[v22][col] += 1
        W[w00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank}, got {col}"
    return U, V, W


def _build_schoolbook(n=2, m=3, p=10):
    """Schoolbook rank n*m*p = 60."""
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


def _try_drop_and_solve(U55, V55, W55, n, m, p, deadline):
    """
    For each term r in the rank-55 decomposition, try to express it as a
    linear combination of the remaining 54 terms. If successful with
    half-integer coefficients, we get a rank-54 decomposition.
    """
    U55 = np.asarray(U55, dtype=np.float64)
    V55 = np.asarray(V55, dtype=np.float64)
    W55 = np.asarray(W55, dtype=np.float64)
    R = 55

    # Build a system of equations from the tensor entries
    # We need: for all (a,b,c), sum_{r != drop} c_r * U[a,r]*V[b,r]*W[c,r] = U[a,drop]*V[b,drop]*W[c,drop]
    # Use random sampling of (a,b,c) triples

    rng = np.random.RandomState(12345)
    n_eq = 400  # number of random equations

    for drop in range(R):
        if time.time() > deadline:
            return None

        keep = [i for i in range(R) if i != drop]
        n_keep = len(keep)

        # Generate random (a,b,c) triples
        a_idx = rng.randint(0, n * m - 1, size=n_eq)
        b_idx = rng.randint(0, m * p - 1, size=n_eq)
        c_idx = rng.randint(0, p * n - 1, size=n_eq)

        # Build A matrix: A[e, r] = U[a_e, keep[r]] * V[b_e, keep[r]] * W[c_e, keep[r]]
        A = np.zeros((n_eq, n_keep), dtype=np.float64)
        b_vec = np.zeros(n_eq, dtype=np.float64)

        for e in range(n_eq):
            a, b, c = a_idx[e], b_idx[e], c_idx[e]
            for r in range(n_keep):
                A[e, r] = U55[a, keep[r]] * V55[b, keep[r]] * W55[c, keep[r]]
            b_vec[e] = U55[a, drop] * V55[b, drop] * W55[c, drop]

        # Solve least squares
        try:
            coeffs, residuals, rk, sv = np.linalg.lstsq(A, b_vec, rcond=None)
        except Exception:
            continue

        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue

        # Check if coefficients are half-integers
        valid = True
        for cv in coeffs:
            if abs(cv * 2 - round(cv * 2)) > 1e-6:
                valid = False
                break
        if not valid:
            continue

        # Build the new decomposition
        U_new = np.zeros((n * m, n_keep), dtype=np.float64)
        V_new = np.zeros((m * p, n_keep), dtype=np.float64)
        W_new = np.zeros((p * n, n_keep), dtype=np.float64)
        for r in range(n_keep):
            U_new[:, r] = coeffs[r] * U55[:, keep[r]]
            V_new[:, r] = coeffs[r] * V55[:, keep[r]]
            W_new[:, r] = coeffs[r] * W55[:, keep[r]]

        if _verify_fast(U_new, V_new, W_new, n, m, p):
            if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()

    return None


def _try_pairwise_merge(U55, V55, W55, n, m, p, deadline):
    """
    Try to merge pairs of terms: replace terms r1 and r2 with a single term
    that is their sum (or difference). This is equivalent to checking if
    U[:,r1]+U[:,r2], V[:,r1]+V[:,r2], W[:,r1]+W[:,r2] is a valid term
    (i.e., the cross terms cancel).
    """
    U55 = np.asarray(U55, dtype=np.float64)
    V55 = np.asarray(V55, dtype=np.float64)
    W55 = np.asarray(W55, dtype=np.float64)
    R = 55

    for r1 in range(R):
        if time.time() > deadline:
            return None
        for r2 in range(r1 + 1, R):
            # Try U_new = U[:,r1] + U[:,r2], etc.
            # The new decomposition has R-1 terms: all terms except r1,r2, plus the merged term
            # But the merged term must satisfy: for all (a,b,c),
            #   (U[a,r1]+U[a,r2])*(V[b,r1]+V[b,r2])*(W[c,r1]+W[c,r2])
            #   = U[a,r1]*V[b,r1]*W[c,r1] + U[a,r2]*V[b,r2]*W[c,r2]
            # This requires cross terms to vanish.
            # Check: U[a,r1]*V[b,r1]*W[c,r2] + U[a,r1]*V[b,r2]*W[c,r1]
            #        + U[a,r2]*V[b,r1]*W[c,r1] + U[a,r2]*V[b,r2]*W[c,r2]... wait
            # Actually: (u1+u2)(v1+v2)(w1+w2) = u1v1w1 + u1v1w2 + u1v2w1 + u1v2w2
            #                                      + u2v1w1 + u2v1w2 + u2v2w1 + u2v2w2
            # We want this to equal u1v1w1 + u2v2w2 (the two original terms).
            # So we need: u1v1w2 + u1v2w1 + u1v2w2 + u2v1w1 + u2v1w2 + u2v2w1 = 0 for all (a,b,c).
            # This is a strong condition. Let's just check numerically.

            # Build the new decomposition
            keep = [i for i in range(R) if i != r1 and i != r2]
            U_new = np.zeros((n * m, R - 1), dtype=np.float64)
            V_new = np.zeros((m * p, R - 1), dtype=np.float64)
            W_new = np.zeros((p * n, R - 1), dtype=np.float64)

            for idx, orig in enumerate(keep):
                U_new[:, idx] = U55[:, orig]
                V_new[:, idx] = V55[:, orig]
                W_new[:, idx] = W55[:, orig]

            # Merged term
            U_new[:, R - 2] = U55[:, r1] + U55[:, r2]
            V_new[:, R - 2] = V55[:, r1] + V55[:, r2]
            W_new[:, R - 2] = W55[:, r1] + W55[:, r2]

            if _verify_fast(U_new, V_new, W_new, n, m, p):
                if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                    return U_new.tolist(), V_new.tolist(), W_new.tolist()

            # Also try difference
            U_new[:, R - 2] = U55[:, r1] - U55[:, r2]
            V_new[:, R - 2] = V55[:, r1] - V55[:, r2]
            W_new[:, R - 2] = W55[:, r1] - W55[:, r2]

            if _verify_fast(U_new, V_new, W_new, n, m, p):
                if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                    return U_new.tolist(), V_new.tolist(), W_new.tolist()

    return None


def _als_polish(U_init, V_init, W_init, target_rank, n, m, p, deadline, n_restarts=10):
    """
    ALS polish: start near the known solution truncated to target_rank,
    run ALS to minimize ||T_reconstructed - T_target||_F.
    """
    U_init = np.asarray(U_init, dtype=np.float64)
    V_init = np.asarray(V_init, dtype=np.float64)
    W_init = np.asarray(W_init, dtype=np.float64)

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_flat = T.reshape(n * m, m * p * p * n)
    T_flat_v = T.reshape(m * p, n * m * p * n)
    T_flat_w = T.reshape(p * n, n * m * m * p)

    best_error = np.inf
    best_U = None
    best_V = None
    best_W = None

    for trial in range(n_restarts):
        if time.time() > deadline:
            break

        np.random.seed(trial * 7919 + target_rank)

        if target_rank <= U_init.shape[1]:
            U_c = U_init[:, :target_rank].copy() + 0.05 * np.random.randn(n * m, target_rank)
            V_c = V_init[:, :target_rank].copy() + 0.05 * np.random.randn(m * p, target_rank)
            W_c = W_init[:, :target_rank].copy() + 0.05 * np.random.randn(p * n, target_rank)
        else:
            U_c = np.random.randn(n * m, target_rank)
            V_c = np.random.randn(m * p, target_rank)
            W_c = np.random.randn(p * n, target_rank)

        for it in range(30):
            if time.time() > deadline:
                break

            # Solve for U: U = T_flat @ (V_c ⊗ W_c)^+
            M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(m * p * p * n, target_rank)
            try:
                U_c = T_flat @ np.linalg.pinv(M.T)
            except Exception:
                break

            # Solve for V
            N = np.einsum('ir,kr->ikr', U_c, W_c).reshape(n * m * p * n, target_rank)
            try:
                V_c = T_flat_v @ np.linalg.pinv(N.T)
            except Exception:
                break

            # Solve for W
            P = np.einsum('ir,jr->ijr', U_c, V_c).reshape(n * m * m * p, target_rank)
            try:
                W_c = T_flat_w @ np.linalg.pinv(P.T)
            except Exception:
                break

        # Check error
        T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
        error = np.linalg.norm(T_check - T, 'fro')

        if error < best_error:
            best_error = error
            best_U = U_c.copy()
            best_V = V_c.copy()
            best_W = W_c.copy()

        if error < 1e-10:
            # Try rounding
            U_r = np.round(U_c * 2) / 2
            V_r = np.round(V_c * 2) / 2
            W_r = np.round(W_c * 2) / 2
            T_check2 = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.linalg.norm(T_check2 - T, 'fro') < 1e-9:
                return U_r.tolist(), V_r.tolist(), W_r.tolist()

    if best_U is not None and best_error < 1e-8:
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        if _verify_fast(U_r, V_r, W_r, n, m, p):
            if _is_valid_entries(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()

    return None


def _random_linear_combination_search(U55, V55, W55, target_rank, n, m, p, deadline, n_trials=200):
    """
    Try random linear combinations of the 55 basis terms to form target_rank
    new terms, then check if the decomposition is valid.
    """
    U55 = np.asarray(U55, dtype=np.float64)
    V55 = np.asarray(V55, dtype=np.float64)
    W55 = np.asarray(W55, dtype=np.float64)
    R = 55

    rng = np.random.RandomState(999)

    for trial in range(n_trials):
        if time.time() > deadline:
            break

        # Randomly select subsets of terms to combine
        U_new = np.zeros((n * m, target_rank), dtype=np.float64)
        V_new = np.zeros((m * p, target_rank), dtype=np.float64)
        W_new = np.zeros((p * n, target_rank), dtype=np.float64)

        for r in range(target_rank):
            # Pick 1-3 random basis terms with random ±1 coefficients
            n_terms = rng.randint(1, 4)
            idx = rng.choice(R, size=n_terms, replace=False)
            for t in idx:
                sign = 1.0 if rng.random() < 0.5 else -1.0
                U_new[:, r] += sign * U55[:, t]
                V_new[:, r] += sign * V55[:, t]
                W_new[:, r] += sign * W55[:, t]

        if _verify_fast(U_new, V_new, W_new, n, m, p):
            if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()

    return None


def _structured_subspace_search(U55, V55, W55, target_rank, n, m, p, deadline, n_trials=100):
    """
    Try to find a target_rank-dimensional subspace of the 55-dimensional
    span that still decomposes the tensor. Use random projections.
    """
    U55 = np.asarray(U55, dtype=np.float64)
    V55 = np.asarray(V55, dtype=np.float64)
    W55 = np.asarray(W55, dtype=np.float64)
    R = 55

    rng = np.random.RandomState(777)

    for trial in range(n_trials):
        if time.time() > deadline:
            break

        # Random target_rank x R matrix C
        C = rng.randn(target_rank, R)
        # Orthonormalize
        Q, _ = np.linalg.qr(C)
        C = Q  # target_rank x R

        U_new = U55 @ C.T  # (n*m, target_rank)
        V_new = V55 @ C.T
        W_new = W55 @ C.T

        # Check if this is a valid decomposition
        # The tensor is sum_r (U[:,r] ⊗ V[:,r] ⊗ W[:,r])
        # = sum_{r,s} C[r,s] (U55[:,s] ⊗ V55[:,s] ⊗ W55[:,s])
        # This is NOT the same as sum_r (U55@C.T)[:,r] ⊗ ... unless C is diagonal.
        # So this approach doesn't directly work. Skip.
        # Instead, try: pick target_rank columns of the 55
        pass

    return None


def _try_rank_reduction(U55, V55, W55, n, m, p, target_rank, deadline):
    """
    Try to reduce from rank 55 to target_rank.
    """
    if target_rank == 54:
        # Try dropping one term
        result = _try_drop_and_solve(U55, V55, W55, n, m, p, deadline)
        if result is not None:
            return result

        # Try pairwise merge
        result = _try_pairwise_merge(U55, V55, W55, n, m, p, deadline)
        if result is not None:
            return result

    # ALS polish
    if time.time() < deadline - 5:
        result = _als_polish(U55, V55, W55, target_rank, n, m, p, deadline, n_restarts=8)
        if result is not None:
            return result

    # Random linear combinations
    if time.time() < deadline - 3:
        result = _random_linear_combination_search(U55, V55, W55, target_rank, n, m, p, deadline, n_trials=100)
        if result is not None:
            return result

    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Build the rank-55 decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    # Verify
    if not (_verify_fast(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Try to reduce rank
    U55 = np.asarray(U_best, dtype=np.float64)
    V55 = np.asarray(V_best, dtype=np.float64)
    W55 = np.asarray(W_best, dtype=np.float64)

    for r in range(54, 49, -1):
        if time.time() > deadline - 15:
            break

        result = _try_rank_reduction(U55, V55, W55, n, m, p, r, deadline - 10)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break

    return {"U": U_best, "V": V_best, "W": W_best}
