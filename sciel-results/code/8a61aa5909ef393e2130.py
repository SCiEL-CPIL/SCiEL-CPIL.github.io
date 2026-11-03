import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]


def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    deadline = time.time() + 135  # Safety margin before 150s kill

    # Build rank-40 schoolbook decomposition
    U40, V40, W40 = _build_schoolbook(n, m, p)
    best_rank = 40
    best_U, best_V, best_W = U40, V40, W40

    # Strategy 1: Linear dependency analysis on the 40 bilinear forms
    # The tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Each rank-1 term contributes u_r ⊗ v_r ⊗ w_r to the tensor
    # Check if the 40 vectors (u_r ⊗ v_r) in R^{(n*m)*(m*p)} are linearly independent
    # If not, we can potentially reduce rank

    # Actually, the dependency must be in the full tensor space
    # Let's check: for each term r, the contribution to T is
    # T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Flatten: T_r as a vector in R^{n*m*p}
    # Check if these 40 vectors are linearly independent

    tensor_dim = n * m * p  # 40
    U_np = np.array(U40, dtype=float)
    V_np = np.array(V40, dtype=float)
    W_np = np.array(W40, dtype=float)

    # Build the 40x40 matrix M where M[r, eq] = T_r[eq]
    M = np.zeros((40, tensor_dim))
    for r in range(40):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * (m * p) + j * p + k
                    M[r, eq_idx] = U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]

    # Check rank of M
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    # If rank_M < 40, there are dependencies

    # For the schoolbook, each (i,j,k) gets exactly one term with value 1
    # So M is a permutation matrix, rank = 40. No dependencies in schoolbook.

    # Strategy 2: Try Strassen-style block decomposition
    # The matrix A is 2x4, B is 4x5, output C is 2x5
    # Split B into 5 columns: B = [B1, B2, B3, B4, B5] each 4x1
    # C = A * B = [A*B1, A*B2, A*B3, A*B4, A*B5]
    # Each A*Bj is a 2x1 vector = 2x4 * 4x1, which has rank 8 (schoolbook)
    # Total: 5 * 8 = 40

    # Can we do better? The 2x4x1 product has rank at most 8.
    # But maybe we can share structure across the 5 products?
    # The A matrix is shared, but the B columns are independent.
    # So the bilinear forms for different k are independent in B.
    # This means we can't easily share across k.

    # Strategy 3: Try to find a rank-36 or lower decomposition using Strassen on 2x2 sub-blocks
    # A is 2x4 = [A1 | A2] where A1, A2 are 2x2
    # B is 4x5 = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2 (2x5)
    # Each A_i * B_i is 2x2 * 2x5, which can be done with Strassen (rank 7 per 2x2x2 block)
    # 2x2 * 2x5: split B_i into two 2x2 + 2x1
    # A_i * [B_i1 | B_i2] where B_i1 is 2x2, B_i2 is 2x1
    # A_i * B_i1: 2x2 * 2x2, rank 7 (Strassen)
    # A_i * B_i2: 2x2 * 2x1, rank 4 (schoolbook)
    # Total per i: 7 + 4 = 11
    # For i=1,2: 2 * 11 = 22
    # But wait, C = A1*B1 + A2*B2, so we need to add the results
    # This doesn't directly give a tensor decomposition

    # Let me think differently. The standard approach:
    # 2x4 * 4x5: 
    # Option A: Treat as 2x(2+2) * (2+2)x5
    # Split A into [A_top; A_bot] (each 1x4) and B into [B_left | B_right] (each 4x2.5?) - no, must be integer splits
    
    # Better: A is 2x4, split into A1 (2x2) and A2 (2x2)
    # B is 4x5, split into B1 (2x5) and B2 (2x5)
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x5. Split B1 into B1a (2x2) and B1b (2x3)
    # A1*B1a: 2x2*2x2, rank 7
    # A1*B1b: 2x2*2x3, rank 6 (schoolbook: 2*2*3=12, but 2x2*2x3 can be done in 6? No, schoolbook is 12)
    # Actually 2x2*2x3: schoolbook is 2*2*3 = 12. Strassen doesn't directly apply.
    # Hmm, this is getting complex.

    # Let me try a different approach: use the known result that rank(2x4x5) <= 36
    # via 4 Strassen 2x2x2 blocks + 1 schoolbook 2x4x1 block
    # This is what the current program does.

    # Strategy 4: Try to find a better decomposition by searching
    # Use random search with structured initialization

    np.random.seed(123)

    # Precompute the equation indices
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all (i,j,k) in [0,2) x [0,4) x [0,5)

    n_eq = n * m * p  # 40

    def verify(U_mat, V_mat, W_mat, R):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try ranks from 35 down to 25
    for R in range(35, 24, -1):
        if time.time() > deadline:
            break

        trials = 500 if R <= 30 else 200

        for trial in range(trials):
            if time.time() > deadline:
                break

            # Random U, V with small integer entries
            U_try = np.random.randint(-1, 2, size=(n * m, R))
            V_try = np.random.randint(-1, 2, size=(m * p, R))

            # Build linear system for W
            n_var = p * n * R
            A_mat = np.zeros((n_eq, n_var))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        var_base = (k * n + i) * R
                        A_mat[eq_idx, var_base:var_base + R] = (
                            U_try[i * m + j, :] * V_try[j * p + k, :]
                        )

            B_vec = np.ones(n_eq)

            try:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, B_vec, rcond=None)
            except:
                continue

            if residuals is None or len(residuals) == 0:
                continue
            if residuals[0] > 1e-10:
                continue

            # Round to half-integers
            W_rounded = np.round(W_flat * 2) / 2.0
            W_mat = W_rounded.reshape((p * n, R))

            if verify(U_try, V_try, W_mat, R):
                best_rank = R
                best_U = U_try.tolist()
                best_V = V_try.tolist()
                best_W = W_mat.tolist()
                break

        if best_rank == R:
            break

    # Strategy 5: Try to reduce the Strassen-36 by removing one column
    # and adjusting the remaining columns
    if best_rank >= 36 and time.time() < deadline:
        # Build the Strassen-36 decomposition
        U36, V36, W36 = _build_strassen_36(n, m, p)
        U36_np = np.array(U36, dtype=float)
        V36_np = np.array(V36, dtype=float)
        W36_np = np.array(W36, dtype=float)

        # Try removing each column and solving for the remaining 35
        for c_remove in range(36):
            if time.time() > deadline:
                break

            U_rem = np.delete(U36_np, c_remove, axis=1)
            V_rem = np.delete(V36_np, c_remove, axis=1)
            W_rem = np.delete(W36_np, c_remove, axis=1)

            R_rem = 35
            n_var = p * n * R_rem
            A_mat = np.zeros((n_eq, n_var))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        var_base = (k * n + i) * R_rem
                        A_mat[eq_idx, var_base:var_base + R_rem] = (
                            U_rem[i * m + j, :] * V_rem[j * p + k, :]
                        )

            B_vec = np.ones(n_eq)

            try:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, B_vec, rcond=None)
            except:
                continue

            if residuals is None or len(residuals) == 0:
                continue
            if residuals[0] > 1e-10:
                continue

            W_rounded = np.round(W_flat * 2) / 2.0
            W_mat = W_rounded.reshape((p * n, R_rem))

            if verify(U_rem, V_rem, W_mat, R_rem):
                best_rank = R_rem
                best_U = U_rem.tolist()
                best_V = V_rem.tolist()
                best_W = W_mat.tolist()
                break

    # Ensure we always return a valid decomposition
    if best_rank == 40:
        best_U, best_V, best_W = U40, V40, W40

    return {"U": best_U, "V": best_V, "W": best_W}


def _build_schoolbook(n, m, p):
    """Build the rank-40 schoolbook decomposition."""
    rank = n * m * p  # 40
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

    assert col == rank
    return U, V, W


def _build_strassen_36(n, m, p):
    """Build Strassen rank-36 decomposition for 2x4 by 4x5."""
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        c = col_start
        # M1 = (A00 + A11) * (B00 + B11)
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M2 = (A10 + A11) * B00
        c = col_start + 1
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        col += 1

        # M3 = A00 * (B01 - B11)
        c = col_start + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M4 = A11 * (B10 - B00)
        c = col_start + 3
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        col += 1

        # M5 = (A00 + A01) * B11
        c = col_start + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        c = col_start + 5
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        c = col_start + 6
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c

    col = add_strassen_222(0, 0, 0, 0)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W
