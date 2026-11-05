import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def get_schoolbook():
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

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get the rank-50 decomposition
    U50, V50, W50 = get_rank50()

    # Verify it's correct
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by finding linear dependencies
    # Key insight: if two columns have the same U[:,r] pattern (up to scalar),
    # we might be able to merge them

    best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()
    best_rank = 50

    # Strategy: try to find columns that can be eliminated
    # For each pair (r, s), check if column r can be expressed as a linear
    # combination of other columns (excluding s) plus a multiple of column s
    # with adjusted W values

    # More practically: check if the tensor can be represented with fewer terms
    # by looking for linear dependencies in the factor matrices

    # First, check simple cases: are any two columns identical in U?
    # If U[:,r] == U[:,s] and V[:,r] == V[:,s], then we can merge them
    # with W[:,merged] = W[:,r] + W[:,s]

    for r in range(50):
        if time.time() > deadline - 10:
            break
        for s in range(r + 1, 50):
            if np.allclose(U50[:, r], U50[:, s], atol=1e-9) and \
               np.allclose(V50[:, r], V50[:, s], atol=1e-9):
                # Can merge these two columns
                U_new = np.delete(U50, [r, s], axis=1)
                V_new = np.delete(V50, [r, s], axis=1)
                W_new = np.delete(W50, [r, s], axis=1)
                # Add merged column
                W_merged = W50[:, r] + W50[:, s]
                U_merged = U50[:, r]
                V_merged = V50[:, r]
                U_new = np.column_stack([U_new, U_merged])
                V_new = np.column_stack([V_new, V_merged])
                W_new = np.column_stack([W_new, W_merged])
                if verify_decomp(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = 49
                    break
        if best_rank < 50:
            break

    # More aggressive: try to eliminate one column at a time
    # by solving for whether the remaining columns can span the tensor
    for r in range(50):
        if time.time() > deadline - 10:
            break
        if best_rank < 50:
            break
        U_test = np.delete(best_U, r, axis=1)
        V_test = np.delete(best_V, r, axis=1)
        W_test = np.delete(best_W, r, axis=1)
        recon = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
        if np.allclose(recon, T, atol=1e-9):
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank -= 1

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
