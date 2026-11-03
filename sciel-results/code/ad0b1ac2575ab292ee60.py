import numpy as np
import time

DIMENSIONS = [4, 4, 8]
_START_TIME = time.time()
DEADLINE = _START_TIME + 130  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def construct():
    n, m, p = DIMENSIONS

    # 1. Define Strassen factors for 2x2x2
    # U2[i, j, r], V2[j, k, r], W2[k, i, r]
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))

    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # 2. Build 4x4x4 factors using tensor product (Kronecker)
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))

    for i in range(4):
        for j in range(4):
            i1, i0 = i // 2, i % 2
            j1, j0 = j // 2, j % 2
            for s in range(7):
                for r in range(7):
                    R = 7 * s + r
                    U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]

    for j in range(4):
        for k in range(4):
            j1, j0 = j // 2, j % 2
            k1, k0 = k // 2, k % 2
            for s in range(7):
                for r in range(7):
                    R = 7 * s + r
                    V4[j, k, R] = V2[j1, k1, s] * V2[j0, k0, r]

    for k in range(4):
        for i in range(4):
            k1, k0 = k // 2, k % 2
            i1, i0 = i // 2, i % 2
            for s in range(7):
                for r in range(7):
                    R = 7 * s + r
                    W4[k, i, R] = W2[k1, i1, s] * W2[k0, i0, r]

    # 3. Build 4x4x8 factors
    # Rank 98
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    # U is duplicated
    U[:, :49] = U4.reshape(16, 49)
    U[:, 49:] = U4.reshape(16, 49)

    # V: First 16 rows (k<4) use first 49 cols. Last 16 rows (k>=4) use last 49 cols.
    # V index: j*8 + k
    for j in range(4):
        for k in range(4):
            idx = j * 8 + k
            V[idx, :49] = V4[j, k, :]

            idx2 = j * 8 + (4 + k)
            V[idx2, 49:] = V4[j, k, :]

    # W: First 16 rows (k<4) use first 49 cols. Last 16 rows (k>=4) use last 49 cols.
    # W index: k*4 + i
    for k in range(4):
        for i in range(4):
            idx = k * 4 + i
            W[idx, :49] = W4[k, i, :]

            idx2 = (4 + k) * 4 + i
            W[idx2, 49:] = W4[k, i, :]

    # 4. Attempt rank reduction by checking for linearly dependent columns
    # that can be eliminated while preserving the tensor identity.
    # We use a simple heuristic: check if any column is a linear combination
    # of others in the sense that its contribution to the tensor is redundant.
    # For exact arithmetic with small integers, we can check if removing a column
    # and solving for its contribution from remaining columns is possible.
    # This is expensive, so we do a limited check.

    if _elapsed() > DEADLINE - 30:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find columns that are linearly dependent in a way that allows removal.
    # We check pairs of columns: if col_a = alpha * col_b in U, V, W simultaneously,
    # we can remove one. But in Strassen, columns are generally independent.
    # Instead, we check if the tensor contribution of one column can be expressed
    # as a sum of contributions of others. This is a system of linear equations.
    # For rank 98, this is 4*4*8 = 128 equations. We can try to find a column r
    # such that T_r = sum_{s != r} c_s T_s.

    # This is a large linear system. We'll use a simplified approach:
    # Check if any column has a very small norm (shouldn't happen with Strassen).
    # Or check for duplicate columns.

    # Check for duplicate columns (exact match)
    U_arr = U.copy()
    V_arr = V.copy()
    W_arr = W.copy()

    remove_set = set()
    for r in range(rank):
        if r in remove_set:
            continue
        # Check if column r is identical to any earlier column
        for s in range(r):
            if s in remove_set:
                continue
            if np.array_equal(U_arr[:, r], U_arr[:, s]) and \
               np.array_equal(V_arr[:, r], V_arr[:, s]) and \
               np.array_equal(W_arr[:, r], W_arr[:, s]):
                remove_set.add(r)
                break

    if remove_set:
        keep = [r for r in range(rank) if r not in remove_set]
        U = U_arr[:, keep]
        V = V_arr[:, keep]
        W = W_arr[:, keep]

    # Check for trivially removable columns (zero in any factor)
    remove_set2 = set()
    for r in range(U.shape[1]):
        if np.allclose(U[:, r], 0, atol=1e-12):
            remove_set2.add(r)
        elif np.allclose(V[:, r], 0, atol=1e-12):
            remove_set2.add(r)
        elif np.allclose(W[:, r], 0, atol=1e-12):
            remove_set2.add(r)

    if remove_set2:
        keep = [r for r in range(U.shape[1]) if r not in remove_set2]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]

    # Verify the decomposition
    n, m, p = DIMENSIONS
    rank_final = U.shape[1]
    U3 = U.reshape(n, m, rank_final)
    V3 = V.reshape(m, p, rank_final)
    W3 = W.reshape(p, n, rank_final)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        # Fallback to original rank 98
        U = np.zeros((16, 98))
        V = np.zeros((32, 98))
        W = np.zeros((32, 98))
        U[:, :49] = U4.reshape(16, 49)
        U[:, 49:] = U4.reshape(16, 49)
        for j in range(4):
            for k in range(4):
                idx = j * 8 + k
                V[idx, :49] = V4[j, k, :]
                idx2 = j * 8 + (4 + k)
                V[idx2, 49:] = V4[j, k, :]
        for k in range(4):
            for i in range(4):
                idx = k * 4 + i
                W[idx, :49] = W4[k, i, :]
                idx2 = (4 + k) * 4 + i
                W[idx2, 49:] = W4[k, i, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
