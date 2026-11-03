"""Matrix multiplication 2x4x5 with rank-36 tensor decomposition.

Uses block decomposition:
  A (2x4) = [A1 | A2], A1, A2 in R^{2x2}
  B (4x5) split into top (2x5) and bottom (2x5)
  B_top = [Bt1 | Bt2 | Bt3] where Bt1, Bt2 are 2x2, Bt3 is 2x1
  B_bot = [Bb1 | Bb2 | Bb3] where Bb1, Bb2 are 2x2, Bb3 is 2x1

  C = [A1*Bt1 + A2*Bb1 | A1*Bt2 + A2*Bb2 | A1*Bt3 + A2*Bb3]

  C1 (2x2) = A1*Bt1 + A2*Bb1  -> 14 rank-1 terms (2x Strassen)
  C2 (2x2) = A1*Bt2 + A2*Bb2  -> 14 rank-1 terms (2x Strassen)
  C3 (2x1) = A1*Bt3 + A2*Bb3  -> 8 rank-1 terms (schoolbook)

  Total: 14 + 14 + 8 = 36

Strategy: Start with the known rank-36 Strassen-based decomposition, then
attempt to find a lower-rank decomposition via:
1. Border-rank perturbation: introduce symbolic epsilon into factor matrices
   and solve for coefficients ensuring tensor identity holds.
2. ALS optimization for candidate lower ranks.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 5]

_START_TIME = time.time()
_DEADLINE = 120.0
_SAFETY_MARGIN = 8.0


def _time_remaining():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)


def _build_strassen_decomp():
    """Build the rank-36 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_strassen(a_row_off, a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        c = start_col
        # M1: (p00+p11)*(q00+q11) -> R00+1, R11+1
        U[0*4 + a_col_off + 0, c] = 1
        U[1*4 + a_col_off + 1, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 0, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 1, c] = 1
        W[(c_col_off+0)*2 + 0, c] = 1
        W[(c_col_off+1)*2 + 1, c] = 1
        c += 1
        # M2: (p10+p11)*(q00+q01) -> R10+1
        U[1*4 + a_col_off + 0, c] = 1
        U[1*4 + a_col_off + 1, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 0, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 1, c] = 1
        W[(c_col_off+0)*2 + 1, c] = 1
        c += 1
        # M3: (p00+p01)*(q10+q11) -> R01+1
        U[0*4 + a_col_off + 0, c] = 1
        U[0*4 + a_col_off + 1, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 0, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 1, c] = 1
        W[(c_col_off+1)*2 + 0, c] = 1
        c += 1
        # M4: p00*(q00-q11) -> R00+1, R11-1
        U[0*4 + a_col_off + 0, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 0, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 1, c] = -1
        W[(c_col_off+0)*2 + 0, c] = 1
        W[(c_col_off+1)*2 + 1, c] = -1
        c += 1
        # M5: p11*(q01-q00) -> R00-1, R01+1
        U[1*4 + a_col_off + 1, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 1, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 0, c] = -1
        W[(c_col_off+0)*2 + 0, c] = -1
        W[(c_col_off+1)*2 + 0, c] = 1
        c += 1
        # M6: (p01+p11)*(q10-q00) -> R11+1
        U[0*4 + a_col_off + 1, c] = 1
        U[1*4 + a_col_off + 1, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 0, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 0, c] = -1
        W[(c_col_off+1)*2 + 1, c] = 1
        c += 1
        # M7: (p00+p10)*(q11-q10) -> R00-1, R10+1
        U[0*4 + a_col_off + 0, c] = 1
        U[1*4 + a_col_off + 0, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 1, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 0, c] = -1
        W[(c_col_off+0)*2 + 0, c] = -1
        W[(c_col_off+0)*2 + 1, c] = 1
        c += 1
        return c

    def add_schoolbook_2x2x1(a_row_off, a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        c = start_col
        U[0*4 + a_col_off + 0, c] = 1
        V[(b_row_off+0)*5 + b_col_off, c] = 1
        W[c_col_off*2 + 0, c] = 1
        c += 1
        U[0*4 + a_col_off + 1, c] = 1
        V[(b_row_off+1)*5 + b_col_off, c] = 1
        W[c_col_off*2 + 0, c] = 1
        c += 1
        U[1*4 + a_col_off + 0, c] = 1
        V[(b_row_off+0)*5 + b_col_off, c] = 1
        W[c_col_off*2 + 1, c] = 1
        c += 1
        U[1*4 + a_col_off + 1, c] = 1
        V[(b_row_off+1)*5 + b_col_off, c] = 1
        W[c_col_off*2 + 1, c] = 1
        c += 1
        return c

    col = add_strassen(0, 0, 0, 0, 0, col)
    col = add_strassen(0, 2, 2, 0, 0, col)
    col = add_strassen(0, 0, 0, 2, 2, col)
    col = add_strassen(0, 2, 2, 2, 2, col)
    col = add_schoolbook_2x2x1(0, 0, 0, 4, 4, col)
    col = add_schoolbook_2x2x1(0, 2, 2, 4, 4, col)

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook():
    """Build the rank-40 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, tol=1e-9):
    """Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_idx = U.reshape(n, m, rank)
    V_idx = V.reshape(m, p, rank)
    W_idx = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
    return np.allclose(T, np.ones((n, m, p)), atol=tol)


def _verify_numeric(U, V, W, tol=1e-9):
    """Verify by computing matrix products."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(12345)
    A_test = rng.randn(n, m)
    B_test = rng.randn(m, p)
    C_expected = A_test @ B_test

    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    U_TA = U.T @ A_flat
    V_TB = V.T @ B_flat

    C_tensor = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            C_tensor[i, k] = np.dot(U_TA * V_TB, W[k * n + i, :])

    return np.allclose(C_tensor, C_expected, atol=tol)


def _als_decompose(target_rank, max_iters=2000, tol=1e-10, seed=42):
    """ALS optimization for tensor decomposition of matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)

    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

    U = rng.randn(n * m, target_rank) * 0.1
    V = rng.randn(m * p, target_rank) * 0.1
    W = rng.reshape if False else rng.randn(p * n, target_rank) * 0.1

    # Precompute index mappings
    # For solving U: for each (i,j), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
    # This is a linear system: for fixed (i,j), define matrix M where
    # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
    # Then U[i*m+j, :] @ M.T = ones(p)
    # i.e., M @ U[i*m+j, :].T = ones(p)

    # For solving V: for each (j,k), define matrix N where
    # N[i, c] = U[i*m+j, c] * W[k*n+i, c]
    # Then N @ V[j*p+k, :].T = ones(n)

    # For solving W: for each (k,i), define matrix O where
    # O[j, c] = U[i*m+j, c] * V[j*p+k, c]
    # Then O @ W[k*n+i, :].T = ones(m)

    def solve_U():
        for i in range(n):
            for j in range(m):
                # M[k, c] = V[j*p+k, c] * W[k*n+i, c] for k in range(p), c in range(target_rank)
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve M @ x = ones(p)
                try:
                    x = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except np.linalg.LinAlgError:
                    x = U[i * m + j, :].copy()
                U[i * m + j, :] = x

    def solve_V():
        for j in range(m):
            for k in range(p):
                # N[i, c] = U[i*m+j, c] * W[k*n+i, c] for i in range(n), c in range(target_rank)
                N = np.zeros((n, target_rank))
                for i in range(n):
                    N[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(N, np.ones(n), rcond=None)[0]
                except np.linalg.LinAlgError:
                    x = V[j * p + k, :].copy()
                V[j * p + k, :] = x

    def solve_W():
        for k in range(p):
            for i in range(n):
                # O[j, c] = U[i*m+j, c] * V[j*p+k, c] for j in range(m), c in range(target_rank)
                O = np.zeros((m, target_rank))
                for j in range(m):
                    O[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x = np.linalg.lstsq(O, np.ones(m), rcond=None)[0]
                except np.linalg.LinAlgError:
                    x = W[k * n + i, :].copy()
                W[k * n + i, :] = x

    prev_error = float('inf')
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break

        solve_U()
        solve_V()
        solve_W()

        # Compute error
        U_idx = U.reshape(n, m, target_rank)
        V_idx = V.reshape(m, p, target_rank)
        W_idx = W.reshape(p, n, target_rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
        error = np.max(np.abs(T - 1.0))

        if error < prev_error * (1 - 1e-6) and it > 10:
            pass  # Still improving
        if error < tol:
            break
        if abs(error - prev_error) < 1e-15:
            # Stuck, try perturbation
            U += rng.randn(*U.shape) * 1e-4
            V += rng.randn(*V.shape) * 1e-4
            W += rng.randn(*W.shape) * 1e-4

        prev_error = error

    return U, V, W, error


def _border_rank_attempt(base_U, base_V, base_W, target_rank, max_iters=500, seed=0):
    """
    Border rank perturbation: start from rank-36 decomp, remove one column,
    then add a small perturbation to all remaining columns and optimize.
    The idea: if rank 36 is a limit of rank 35, then removing one column and
    adjusting the others should still yield a valid decomposition.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)

    # Remove one column (try different ones)
    # Actually, let's try a different approach: start with rank-36, then
    # try to express it as rank-35 by finding a linear dependency.
    # If column c can be written as a linear combination of others in all three
    # matrices simultaneously, we can eliminate it.

    # For each column c, check if U[:,c] is in span of other U columns,
    # V[:,c] is in span of other V columns, and W[:,c] is in span of other W columns.
    # This is unlikely for a minimal decomposition, but let's try.

    # Alternative: use the structure. The rank-36 decomp has:
    # 4 Strassen blocks (7 terms each = 28) + 2 schoolbook blocks (4 terms each = 8) = 36
    # Can we merge the two schoolbook blocks (8 terms) into fewer?
    # C3 = A1*Bt3 + A2*Bb3
    # A1 is 2x2, Bt3 is 2x1, A2 is 2x2, Bb3 is 2x1
    # C3[i] = A1[i,0]*Bt3[0] + A1[i,1]*Bt3[1] + A2[i,0]*Bb3[0] + A2[i,1]*Bb3[1]
    # = A1[i,0]*Bt3[0] + A1[i,1]*Bt3[1] + A[i,2]*B[2,4] + A[i,3]*B[3,4]
    # This is 4 terms per row, 8 total. Can we do better?
    # For 2x1 output, we need at least... the bilinear complexity of
    # computing 2 linear forms each involving 4 products.
    # Each output C[i] = sum of 4 products. Two outputs share no products.
    # So we need at least 4+4=8? Unless we can share computation.
    # C[0] = A[0,0]*B[0,4] + A[0,1]*B[1,4] + A[0,2]*B[2,4] + A[0,3]*B[3,4]
    # C[1] = A[1,0]*B[0,4] + A[1,1]*B[1,4] + A[1,2]*B[2,4] + A[1,3]*B[3,4]
    # These are independent bilinear forms. The rank of the tensor for
    # (2x2 * 2x1) + (2x2 * 2x1) is at most 8 (schoolbook) and at least...
    # Actually, let's see if we can use Strassen-like tricks.
    # Let X = A1[:,0] + A1[:,1] (2x1), Y = Bt3[0] + Bt3[1] (scalar)
    # Then X*Y = A1[0,0]*Bt3[0] + A1[0,1]*Bt3[0] + A1[1,0]*Bt3[0] + A1[1,1]*Bt3[0]
    #           + A1[0,0]*Bt3[1] + A1[0,1]*Bt3[1] + A1[1,0]*Bt3[1] + A1[1,1]*Bt3[1]
    # This doesn't directly help since we need specific products.

    # Let's try a different structural approach:
    # Instead of 2 Strassen (14) + 2 Strassen (14) + schoolbook (8) = 36,
    # can we use a different partition?

    # Alternative: treat the whole problem differently.
    # A is 2x4, B is 4x5.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]

    # Group: (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    # First group: 2x2 * 2x5 -> but B[0:2, :] is 2x5, not 2x2.
    # Let's stick with the current partition but try to optimize the schoolbook part.

    # For now, let's just try ALS at various ranks.
    return None


def construct():
    """
    Find the lowest-rank exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS

    # Start with known rank-36 decomposition
    U36, V36, W36 = _build_strassen_decomp()

    # Verify it
    if not _verify(U36, V36, W36):
        # Fall back to schoolbook
        U40, V40, W40 = _build_schoolbook()
        return {"U": U40.tolist(), "V": V40.tolist(), "W": W40.tolist()}

    best_rank = 36
    best_U, best_V, best_W = U36, V36, W36

    # Try to find lower rank via ALS
    # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But realistically, for 2x4*4x5, the rank is probably around 24-30.
    # Let's try ranks 24 through 35.

    for target_rank in range(24, 36):
        if _time_remaining() < 10.0:
            break

        # Try multiple random seeds
        for seed in range(5):
            if _time_remaining() < 5.0:
                break

            U_try, V_try, W_try, error = _als_decompose(
                target_rank, max_iters=1000, tol=1e-10, seed=seed
            )

            if _time_remaining() < 3.0:
                break

            if error < 1e-8:
                # Verify more carefully
                if _verify(U_try, V_try, W_try, tol=1e-8) and _verify_numeric(U_try, V_try, W_try, tol=1e-8):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_try, V_try, W_try
                    break  # Found valid decomposition at this rank, try lower

        # If we found a valid decomp at this rank, try lower ranks
        if best_rank == target_rank:
            continue
        else:
            # Didn't find at this rank, continue to next
            continue

    # Also try a structured approach: modify the rank-36 decomp to see if
    # we can eliminate one column by adjusting others.
    if _time_remaining() > 15.0:
        # Try to find if any column in the rank-36 decomp is linearly dependent
        # on the others (in all three matrices simultaneously).
        # This is a very strong condition and unlikely, but let's check.
        for col_to_remove in range(36):
            if _time_remaining() < 10.0:
                break

            # Remove column col_to_remove
            mask = [i for i in range(36) if i != col_to_remove]
            U_reduced = U36[:, mask]
            V_reduced = V36[:, mask]
            W_reduced = W36[:, mask]

            # Now try to "absorb" the removed column into the remaining ones
            # by adding a small perturbation and re-optimizing.
            # This is essentially ALS starting from a near-rank-35 point.

            # Add the removed column's contribution as a perturbation
            eps = 1e-3
            U_pert = U_reduced.copy()
            V_pert = V_reduced.copy()
            W_pert = W_reduced.copy()

            # Try ALS from this starting point
            # Actually, let's just run ALS at rank 35 with this initialization
            U_init = U_reduced + np.random.RandomState(0).randn(*U_reduced.shape) * eps
            V_init = V_reduced + np.random.RandomState(1).randn(*V_reduced.shape) * eps
            W_init = W_reduced + np.random.RandomState(2).randn(*W_reduced.shape) * eps

            # Run ALS
            n, m, p = DIMENSIONS
            target_rank = 35
            rng = np.random.RandomState(42)

            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()

            for it in range(500):
                if _time_remaining() < 8.0:
                    break

                # Solve U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V_cur[j * p + k, :] * W_cur[k * n + i, :]
                        try:
                            x = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        except np.linalg.LinAlgError:
                            x = U_cur[i * m + j, :].copy()
                        U_cur[i * m + j, :] = x

                # Solve V
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, target_rank))
                        for i in range(n):
                            N[i, :] = U_cur[i * m + j, :] * W_cur[k * n + i, :]
                        try:
                            x = np.linalg.lstsq(N, np.ones(n), rcond=None)[0]
                        except np.linalg.LinAlgError:
                            x = V_cur[j * p + k, :].copy()
                        V_cur[j * p + k, :] = x

                # Solve W
                for k in range(p):
                    for i in range(n):
                        O = np.zeros((m, target_rank))
                        for j in range(m):
                            O[j, :] = U_cur[i * m + j, :] * V_cur[j * p + k, :]
                        try:
                            x = np.linalg.lstsq(O, np.ones(m), rcond=None)[0]
                        except np.linalg.LinAlgError:
                            x = W_cur[k * n + i, :].copy()
                        W_cur[k * n + i, :] = x

                # Check convergence
                U_idx = U_cur.reshape(n, m, target_rank)
                V_idx = V_cur.reshape(m, p, target_rank)
                W_idx = W_cur.reshape(p, n, target_rank)
                T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
                error = np.max(np.abs(T - 1.0))

                if error < 1e-8:
                    if _verify(U_cur, V_cur, W_cur, tol=1e-8) and _verify_numeric(U_cur, V_cur, W_cur, tol=1e-8):
                        best_rank = 35
                        best_U, best_V, best_W = U_cur, V_cur, W_cur
                    break

    # Final verification
    if not _verify(best_U, best_V, best_W, tol=1e-8):
        # Fall back to rank-36
        best_U, best_V, best_W = U36, V36, W36
        best_rank = 36

    if not _verify(best_U, best_V, best_W, tol=1e-8):
        # Ultimate fallback: schoolbook
        U40, V40, W40 = _build_schoolbook()
        best_U, best_V, best_W = U40, V40, W40
        best_rank = 40

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
