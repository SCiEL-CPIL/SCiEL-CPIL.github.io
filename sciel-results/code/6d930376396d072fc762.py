"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Strategy:
  1. Build the known rank-55 Strassen-based decomposition (baseline).
  2. Try to reduce rank using ALS (alternating least squares) optimization
     starting from rank-55, targeting rank 54.
  3. If ALS finds a valid decomposition with integer/half-integer entries,
     return it. Otherwise return the rank-55 baseline.

The key insight is that the tensor has structure: the 2x3 matrix has 3 columns,
and the 3x10 matrix has 3 rows. By splitting into j=0 (scalar) and j=1,2 (2x2 Strassen),
we get rank 20 + 35 = 55. The question is whether rank 54 or lower is achievable.
"""

import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using vectorized numpy operations."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    # Reconstruct tensor via outer product sum
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T = np.einsum('ar,br,cr->abc', U, V, W)

    # Check nonzero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False

    # Check zero entries via random sampling
    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero_set.add((i * m + j, j * p + k, k * n + i))

    for _ in range(500):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        if (a, b, c) not in nonzero_set:
            if abs(T[a, b, c]) > 1e-9:
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


def _round_to_half(U, V, W):
    """Round all entries to nearest half-integer."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    U = np.round(U * 2) / 2.0
    V = np.round(V * 2) / 2.0
    W = np.round(W * 2) / 2.0
    return U, V, W


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.

    A is 2x3: columns j=0,1,2.
    B is 3x10: rows j=0,1,2.
    C is 2x10: C[i,k] = sum_j A[i,j]*B[j,k].

    Part A: j=0 -> A[i,0]*B[0,k] for i in {0,1}, k in 0..9.  20 terms.
    Part B: j=1,2 -> A2 (2x2) * B2 (2x10). Split B2 into 5 blocks of 2 columns.
             Each 2x2 * 2x2 block uses Strassen: 7 multiplications. 5*7 = 35.
    Total: 55.
    """
    rank = 55
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
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
        U[u_11, col] += 1
        U[u_22, col] += 1
        V[v_11, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += 1
        W[w_11, col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21, col] += 1
        U[u_22, col] += 1
        V[v_11, col] += 1
        W[w_10, col] += 1
        W[w_11, col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11, col] += 1
        V[v_12, col] += 1
        V[v_22, col] += -1
        W[w_01, col] += 1
        W[w_11, col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22, col] += 1
        V[v_21, col] += 1
        V[v_11, col] += -1
        W[w_00, col] += 1
        W[w_10, col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11, col] += 1
        U[u_12, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += -1
        W[w_01, col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21, col] += 1
        U[u_11, col] += -1
        V[v_11, col] += 1
        V[v_12, col] += 1
        W[w_11, col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12, col] += 1
        U[u_22, col] += -1
        V[v_21, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U.tolist(), V.tolist(), W.tolist()


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
    return U.tolist(), V.tolist(), W.tolist()


def _als_reduce_rank(U, V, W, target_rank, n, m, p, max_iters=5000, tol=1e-12, deadline=None):
    """
    Try to reduce rank via ALS optimization.
    
    Start with rank-55 matrices, drop one column (to get target_rank),
    then optimize all three matrices to minimize reconstruction error.
    
    Returns (U, V, W) as lists if successful, else None.
    """
    n_factors = len(U)  # n*m for U, m*p for V, p*n for W
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    orig_rank = U.shape[1]
    
    # Drop the last column to get target_rank
    U = U[:, :target_rank]
    V = V[:, :target_rank]
    W = W[:, :target_rank]
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def _compute_error():
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T_recon - T)
    
    # ALS iterations
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each row a, solve min ||T[:,b,c] - U[a,:] * (V⊙W)[b,c,:]||
        # V⊙W has shape (m*p, p*n, rank)
        VW = np.einsum('br,cr->bc r'.replace(' ', ''), V, W)
        # Actually: VW[b,c,r] = V[b,r]*W[c,r]
        VW = np.einsum('br,cr->bcr', V, W)
        
        for a in range(n * m):
            # Target: T[a,b,c] for all b,c
            # We want U[a,:] such that sum_r U[a,r]*VW[b,c,r] ≈ T[a,b,c]
            # Reshape VW to (m*p*p*n, rank) and T[a] to (m*p*p*n,)
            VW_flat = VW.reshape(-1, target_rank)
            T_a_flat = T[a].reshape(-1)
            # Solve least squares: VW_flat @ U[a,:] ≈ T_a_flat
            # U[a,:] = (VW_flat^T VW_flat)^{-1} VW_flat^T T_a_flat
            try:
                U[a, :] = np.linalg.lstsq(VW_flat, T_a_flat, rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
        
        # Update V similarly
        UW = np.einsum('ar,cr->acr', U, W)  # UW[a,c,r] = U[a,r]*W[c,r]
        for b in range(m * p):
            UW_flat = UW.reshape(-1, target_rank)
            T_b_flat = T[:, b, :].reshape(-1)
            try:
                V[b, :] = np.linalg.lstsq(UW_flat, T_b_flat, rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
        
        # Update W
        UV = np.einsum('ar,br->abr', U, V)  # UV[a,b,r] = U[a,r]*V[b,r]
        for c in range(p * n):
            UV_flat = UV.reshape(-1, target_rank)
            T_c_flat = T[:, :, c].reshape(-1)
            try:
                W[c, :] = np.linalg.lstsq(UV_flat, T_c_flat, rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
        
        # Check convergence
        if it % 50 == 0:
            err = _compute_error()
            if err < tol:
                break
    
    err = _compute_error()
    if err < 1e-9:
        return U.tolist(), V.tolist(), W.tolist()
    return None


def _random_search_rank(n, m, p, rank, values, trials, deadline=None):
    """Random search for a decomposition of the given rank."""
    for t in range(trials):
        if deadline is not None and time.time() > deadline:
            break
        U = [[random.choice(values) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(values) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(values) for _ in range(rank)] for _ in range(p * n)]
        if _verify_fast(U, V, W, n, m, p) and _is_valid_entries(U, V, W):
            return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    # Verify it
    if not (_verify_fast(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        # Fallback to schoolbook
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Try ALS reduction to rank 54
    if time.time() < deadline - 60:
        result = _als_reduce_rank(U_best, V_best, W_best, 54, n, m, p,
                                   max_iters=3000, tol=1e-12, deadline=deadline - 60)
        if result is not None:
            U_r, V_r, W_r = result
            # Round to half-integers and verify
            U_arr, V_arr, W_arr = _round_to_half(U_r, V_r, W_r)
            U_r = U_arr.tolist()
            V_r = V_arr.tolist()
            W_r = W_arr.tolist()
            if _verify_fast(U_r, V_r, W_r, n, m, p) and _is_valid_entries(U_r, V_r, W_r):
                U_best, V_best, W_best = U_r, V_r, W_r
                best_rank = 54

    # If still at 55, try random search for rank 53, 54
    if best_rank > 54 and time.time() < deadline - 20:
        values = [0, 0.5, 1, -0.5, -1]
        for r in range(53, best_rank):
            if time.time() > deadline - 10:
                break
            result = _random_search_rank(n, m, p, r, values, trials=200, deadline=deadline - 10)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = r
                break

    return {"U": U_best, "V": V_best, "W": W_best}
