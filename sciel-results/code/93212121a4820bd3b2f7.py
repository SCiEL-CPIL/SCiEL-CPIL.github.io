"""Rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
Uses block Strassen (rank 22) as baseline, then searches for rank-21 via
targeted column-merge optimization within Strassen blocks.
"""
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 135.0  # seconds, leaves 15s margin before 150s kill


def _elapsed():
    return time.time() - _T0


def _remaining():
    return _DEADLINE - _elapsed()


def _verify_full(U, V, W, n, m, p, tol=1e-6):
    """Full bilinear verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (ia == i and ja == jb and kb == k) else 0.0
                            Uidx = ia * m + ja
                            Vidx = jb * p + kb
                            Widx = k * n + i
                            actual = np.dot(W[Widx, :], U[Uidx, :] * V[Vidx, :])
                            if abs(actual - expected) > tol:
                                return False
    return True


def _bilinear_error_fast(U, V, W, n, m, p):
    """Compute squared bilinear error efficiently using tensor contractions."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Build the full tensor T[i,j,k, ia, ja, jb, kb] = sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r]
    # Expected: delta(i,ia)*delta(j,jb)*delta(k,kb)
    
    total_err = 0.0
    for ia in range(n):
        for ja in range(m):
            U_row = U[ia * m + ja, :]
            for jb in range(m):
                for kb in range(p):
                    V_row = V[jb * p + kb, :]
                    prod = U_row * V_row
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (ia == i and ja == jb and kb == k) else 0.0
                            W_row = W[k * n + i, :]
                            actual = np.dot(W_row, prod)
                            total_err += (actual - expected) ** 2
    return total_err


def _rank22_list(n, m, p):
    """Construct rank-22 decomposition using block Strassen."""
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    # Block 1: 7 Strassen terms for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A: 0=A00, 1=A01, 3=A10, 4=A11
    # B: 0=B00, 1=B01, 4=B10, 5=B11
    # W: 0=C00, 2=C01, 1=C10, 3=C11

    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1

    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1

    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1

    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1

    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1

    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1

    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1

    # Block 2: 7 Strassen terms for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # B: 2=B02, 3=B03, 6=B12, 7=B13
    # W: 4=C02, 6=C03, 5=C12, 7=C13

    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1

    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1

    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1

    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[5][col] = 1
    col += 1

    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1

    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1

    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1

    # Block 3: 8 schoolbook terms for A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    # A: 2=A02, 5=A12
    # B: 8=B20, 9=B21, 10=B22, 11=B23
    for i, ai in [(0, 2), (1, 5)]:
        for k, bk in [(0, 8), (1, 9), (2, 10), (3, 11)]:
            U[ai][col] = 1
            V[bk][col] = 1
            W[k * n + i][col] = 1
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W


def _schoolbook(n, m, p):
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
    return U, V, W


def _to_dict(U, V, W):
    return {"U": U, "V": V, "W": W}


def _try_merge_columns(U22, V22, W22, col_a, col_b, n, m, p, max_iter=500):
    """Try to merge two columns into one by optimizing the merged column and adjusting others."""
    r = 21
    nm, mp, pn = n * m, m * p, p * n
    
    # Remove both columns, create a new merged column
    keep = [c for c in range(22) if c != col_a and c != col_b]
    U_init = np.array(U22, dtype=float)[:, keep].copy()
    V_init = np.array(V22, dtype=float)[:, keep].copy()
    W_init = np.array(W22, dtype=float)[:, keep].copy()
    
    # Initialize merged column as sum of the two
    U_merged = np.array(U22, dtype=float)[:, col_a] + np.array(U22, dtype=float)[:, col_b]
    V_merged = np.array(V22, dtype=float)[:, col_a] + np.array(V22, dtype=float)[:, col_b]
    W_merged = np.array(W22, dtype=float)[:, col_a] + np.array(W22, dtype=float)[:, col_b]
    
    # Append merged column
    U_init = np.column_stack([U_init, U_merged])
    V_init = np.column_stack([V_init, V_merged])
    W_init = np.column_stack([W_init, W_merged])
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def objective(x):
        U = x[:nm * r].reshape(nm, r)
        V = x[nm * r:nm * r + mp * r].reshape(mp, r)
        W = x[nm * r + mp * r:].reshape(pn, r)
        return _bilinear_error_fast(U, V, W, n, m, p)
    
    try:
        from scipy.optimize import minimize
        res = minimize(objective, x0, method='BFGS',
                       options={'maxiter': max_iter, 'gtol': 1e-7})
        if res.fun < 1e-10:
            x = res.x
            U = x[:nm * r].reshape(nm, r)
            V = x[nm * r:nm * r + mp * r].reshape(mp, r)
            W = x[nm * r + mp * r:].reshape(pn, r)
            if _verify_full(U, V, W, n, m, p, tol=1e-5):
                return U.tolist(), V.tolist(), W.tolist()
    except Exception:
        pass
    return None


def _try_drop_column(U22, V22, W22, drop_col, n, m, p, max_iter=300):
    """Try dropping one column and optimizing the rest."""
    r = 21
    nm, mp, pn = n * m, m * p, p * n
    
    keep = [c for c in range(22) if c != drop_col]
    U_init = np.array(U22, dtype=float)[:, keep].copy()
    V_init = np.array(V22, dtype=float)[:, keep].copy()
    W_init = np.array(W22, dtype=float)[:, keep].copy()
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def objective(x):
        U = x[:nm * r].reshape(nm, r)
        V = x[nm * r:nm * r + mp * r].reshape(mp, r)
        W = x[nm * r + mp * r:].reshape(pn, r)
        return _bilinear_error_fast(U, V, W, n, m, p)
    
    try:
        from scipy.optimize import minimize
        res = minimize(objective, x0, method='BFGS',
                       options={'maxiter': max_iter, 'gtol': 1e-7})
        if res.fun < 1e-10:
            x = res.x
            U = x[:nm * r].reshape(nm, r)
            V = x[nm * r:nm * r + mp * r].reshape(mp, r)
            W = x[nm * r + mp * r:].reshape(pn, r)
            if _verify_full(U, V, W, n, m, p, tol=1e-5):
                return U.tolist(), V.tolist(), W.tolist()
    except Exception:
        pass
    return None


def construct():
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n

    # Start with rank-22
    U22, V22, W22 = _rank22_list(n, m, p)
    
    # Verify baseline
    if not _verify_full(U22, V22, W22, n, m, p):
        return _to_dict(*_schoolbook(n, m, p))

    if _remaining() < 30.0:
        return _to_dict(U22, V22, W22)

    # Try to find rank-21 by merging pairs of columns from the same Strassen block
    # Block 1: columns 0-6, Block 2: columns 7-13, Block 3: columns 14-21
    
    # First, try merging pairs within the same Strassen block (more likely to succeed)
    # Pairs in Block 1 that share structure:
    merge_pairs = [
        # Block 1 pairs (columns 0-6)
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 4), (3, 5), (3, 6),
        (4, 5), (4, 6),
        (5, 6),
        # Block 2 pairs (columns 7-13)
        (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13),
        (8, 9), (8, 10), (8, 11), (8, 12), (8, 13),
        (9, 10), (9, 11), (9, 12), (9, 13),
        (10, 11), (10, 12), (10, 13),
        (11, 12), (11, 13),
        (12, 13),
    ]
    
    for col_a, col_b in merge_pairs:
        if _remaining() < 20.0:
            break
        result = _try_merge_columns(U22, V22, W22, col_a, col_b, n, m, p, max_iter=200)
        if result is not None:
            return _to_dict(*result)

    # If merge didn't work, try dropping single columns
    for drop_col in range(22):
        if _remaining() < 15.0:
            break
        result = _try_drop_column(U22, V22, W22, drop_col, n, m, p, max_iter=150)
        if result is not None:
            return _to_dict(*result)

    # Fall back to rank-22
    return _to_dict(U22, V22, W22)
