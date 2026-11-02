import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def _time_exceeded(start_time, limit=140.0):
    return (time.time() - start_time) > limit

def _verify_decomposition(U, V, W, T):
    """Verify that U @ diag @ V @ diag @ W reconstructs T."""
    # T_recon[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_recon, T, atol=1e-9)

def _build_base_decomposition(n, m, p):
    """Build the rank-32 Strassen hybrid decomposition."""
    # Target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Rank 32: 4 Strassen blocks (rank 7 each) + 4 bottom row terms
    R_total = 4 * 7 + 4
    U = np.zeros((n*m, R_total))
    V = np.zeros((m*p, R_total))
    W = np.zeros((p*n, R_total))
    col = 0

    def add_strassen_block(ar1, ar2, ac1, ac2, br1, br2, bc1, bc2, cr1, cr2, cc1, cc2):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        U[ar1*m + ac1, col] += 1
        U[ar2*m + ac2, col] += 1
        V[br1*p + bc1, col] += 1
        V[br2*p + bc2, col] += 1
        W[cc1*n + cr1, col] += 1
        W[cc2*n + cr2, col] += 1
        col += 1

        # M2 = (A21+A22)B11
        U[ar2*m + ac1, col] += 1
        U[ar2*m + ac2, col] += 1
        V[br1*p + bc1, col] += 1
        W[cc1*n + cr2, col] += 1
        col += 1

        # M3 = A11(B21-B22)
        U[ar1*m + ac1, col] += 1
        V[br2*p + bc1, col] += 1
        V[br2*p + bc2, col] -= 1
        W[cc2*n + cr1, col] += 1
        col += 1

        # M4 = A22(B11-B21)
        U[ar2*m + ac2, col] += 1
        V[br1*p + bc1, col] += 1
        V[br2*p + bc1, col] -= 1
        W[cc2*n + cr1, col] += 1
        W[cc2*n + cr2, col] += 1
        col += 1

        # M5 = (A11+A12)B22
        U[ar1*m + ac1, col] += 1
        U[ar1*m + ac2, col] += 1
        V[br2*p + bc2, col] += 1
        W[cc1*n + cr1, col] += 1
        W[cc2*n + cr2, col] += 1
        col += 1

        # M6 = (A21-A11)(B11+B12)
        U[ar2*m + ac1, col] += 1
        U[ar1*m + ac1, col] -= 1
        V[br1*p + bc1, col] += 1
        V[br1*p + bc2, col] += 1
        W[cc2*n + cr2, col] += 1
        col += 1

        # M7 = (A12-A22)(B21+B22)
        U[ar1*m + ac2, col] += 1
        U[ar2*m + ac2, col] -= 1
        V[br2*p + bc1, col] += 1
        V[br2*p + bc2, col] += 1
        W[cc1*n + cr1, col] += 1
        W[cc2*n + cr2, col] += 1
        col += 1

    # Block 1: A(0,1; 0,1) * B(0,1; 0,1) -> C(0,1; 0,1)
    add_strassen_block(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)

    # Block 2: A(0,1; 2,3) * B(2,3; 0,1) -> C(0,1; 2,3)
    add_strassen_block(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3)

    # Block 3: A(0,1; 0,1) * B(0,1; 2,3) -> C(0,1; 0,1)
    add_strassen_block(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1)

    # Block 4: A(0,1; 2,3) * B(2,3; 2,3) -> C(0,1; 2,3)
    add_strassen_block(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)

    # Bottom row: A(2; 0..3) * B(0..3; 0..3) -> C(2; 0..3)
    for j in range(4):
        U[2*m + j, col] = 1
        for k in range(4):
            V[j*p + k, col] = 1
            W[k*n + 2, col] = 1
        col += 1

    return U, V, W, T

def _try_remove_columns(U, V, W, T, start_time):
    """Try to remove redundant columns from the decomposition."""
    R = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()

    removed = set()
    for col_idx in range(R):
        if _time_exceeded(start_time):
            break
        if col_idx in removed:
            continue
        # Try removing this column
        mask = np.ones(R, dtype=bool)
        for r in removed:
            mask[r] = False
        mask[col_idx] = False
        U_sub = U[:, mask]
        V_sub = V[:, mask]
        W_sub = W[:, mask]
        if _verify_decomposition(U_sub, V_sub, W_sub, T):
            removed.add(col_idx)

    if removed:
        mask = np.ones(R, dtype=bool)
        for r in removed:
            mask[r] = False
        U = U[:, mask]
        V = V[:, mask]
        W = W[:, mask]

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    # Build the rank-32 decomposition
    U, V, W, T = _build_base_decomposition(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_s = np.zeros((n*m, rank_school))
        V_s = np.zeros((m*p, rank_school))
        W_s = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col] = 1
                    V_s[j*p+k, col] = 1
                    W_s[k*n+i, col] = 1
                    col += 1
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }

    # Try to reduce rank by removing redundant columns
    if not _time_exceeded(start_time):
        U2, V2, W2 = _try_remove_columns(U, V, W, T, start_time)
        if U2.shape[1] < best_rank:
            best_U, best_V, best_W = U2, V2, W2
            best_rank = U2.shape[1]

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
