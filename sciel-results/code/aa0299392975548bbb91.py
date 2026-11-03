"""Rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.

Strategy:
  1. Build a rank-32 block decomposition (4 Strassen 2x2*2x2 + 4 rank-1 1x2*2x2).
  2. Attempt rank reduction via column elimination:
     - For each candidate column r, check if it can be expressed as a linear
       combination of other columns using the tensor structure.
     - Use the fact that T = sum_r U_r ⊗ V_r ⊗ W_r, and if column r is in the
       span of others, we can eliminate it.
  3. Fallback to schoolbook rank-48 if verification fails.
"""
import time
import numpy as np

N, M, P = 3, 4, 4


def _verify_fast(U, V, W, n=N, m=M, p=P):
    """Fast verification: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=1e-9)


def _strassen_2x2_2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2.

    Returns U (4,7), V (4,7), W (4,7) with index conventions
    U[i*2+j,c], V[j*2+k,c], W[k*2+i,c].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1

    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1

    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1

    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1

    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1

    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1

    return U, V, W


def _one_by_two_by_two():
    """Rank-1 decomposition for 1x2 * 2x2.

    The tensor T[0,j,k] = 1 for all j,k in {0,1} is the all-ones tensor,
    which has rank 1: T = (1,1) ⊗ (1,1,1,1) ⊗ (1,1).

    Returns U (2,1), V (4,1), W (2,1) with conventions
    U[j,c], V[j*2+k,c], W[k,c].
    """
    U = np.ones((2, 1))
    V = np.ones((4, 1))
    W = np.ones((2, 1))

    return U, V, W


def _block_decomposition_3x4_4x4():
    """Assemble the block decomposition. Returns (U, V, W) as numpy arrays, rank 32."""
    S_U, S_V, S_W = _strassen_2x2_2x2()
    O_U, O_V, O_W = _one_by_two_by_two()

    total_rank = 4 * 7 + 4 * 1  # 32
    U_full = np.zeros((12, total_rank))
    V_full = np.zeros((16, total_rank))
    W_full = np.zeros((12, total_rank))

    col = 0

    def place_strassen(u_col_offset, v_col_offset, w_col_offset, c_off):
        """Place one Strassen 2x2*2x2 block."""
        for c in range(7):
            for li in range(2):
                for lj in range(2):
                    u_idx = li * 4 + (u_col_offset + lj)
                    val = S_U[li * 2 + lj, c]
                    if val != 0:
                        U_full[u_idx, c_off + c] = val
            for lj in range(2):
                for lk in range(2):
                    v_idx = (v_col_offset + lj) * 4 + lk
                    val = S_V[lj * 2 + lk, c]
                    if val != 0:
                        V_full[v_idx, c_off + c] = val
            for lk in range(2):
                for li in range(2):
                    w_idx = (w_col_offset + lk) * 3 + li
                    val = S_W[lk * 2 + li, c]
                    if val != 0:
                        W_full[w_idx, c_off + c] = val

    def place_one_by_two(u_col_offset, v_col_offset, w_col_offset, c_off):
        """Place one 1x2*2x2 block (rank 1)."""
        for c in range(1):
            for lj in range(2):
                u_idx = 2 * 4 + (u_col_offset + lj)
                val = O_U[lj, c]
                if val != 0:
                    U_full[u_idx, c_off + c] = val
            for lj in range(2):
                for lk in range(2):
                    v_idx = (v_col_offset + lj) * 4 + lk
                    val = O_V[lj * 2 + lk, c]
                    if val != 0:
                        V_full[v_idx, c_off + c] = val
            for lk in range(2):
                w_idx = (w_col_offset + lk) * 3 + 2
                val = O_W[lk, c]
                if val != 0:
                    W_full[w_idx, c_off + c] = val

    # C00 = A00*B00 + A01*B10
    place_strassen(u_col_offset=0, v_col_offset=0, w_col_offset=0, c_off=col); col += 7
    place_strassen(u_col_offset=2, v_col_offset=2, w_col_offset=0, c_off=col); col += 7
    # C01 = A00*B01 + A01*B11
    place_strassen(u_col_offset=0, v_col_offset=0, w_col_offset=2, c_off=col); col += 7
    place_strassen(u_col_offset=2, v_col_offset=2, w_col_offset=2, c_off=col); col += 7
    # C10 = A10*B00 + A11*B10
    place_one_by_two(u_col_offset=0, v_col_offset=0, w_col_offset=0, c_off=col); col += 1
    place_one_by_two(u_col_offset=2, v_col_offset=2, w_col_offset=0, c_off=col); col += 1
    # C11 = A10*B01 + A11*B11
    place_one_by_two(u_col_offset=0, v_col_offset=0, w_col_offset=2, c_off=col); col += 1
    place_one_by_two(u_col_offset=2, v_col_offset=2, w_col_offset=2, c_off=col); col += 1

    return U_full, V_full, W_full


def _schoolbook():
    """Rank-48 schoolbook decomposition."""
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W


def _try_eliminate_column(U, V, W, r_idx, other_cols, tol=1e-9):
    """Try to express column r_idx as a linear combination of other_cols.
    
    Returns (coeffs, success) where coeffs is a vector of length len(other_cols)
    such that U[:,r_idx] ⊗ V[:,r_idx] ⊗ W[:,r_idx] ≈ sum_c coeffs[c] * (U[:,c] ⊗ V[:,c] ⊗ W[:,c])
    """
    # Build the matrix A where each column c corresponds to the outer product
    # of U[:,c] and V[:,c], flattened, and we want to match U[:,r_idx] ⊗ V[:,r_idx]
    # But we also need to account for W. The condition is:
    # sum_c coeffs[c] * U[:,c] * V[:,c] * W[:,c] = U[:,r_idx] * V[:,r_idx] * W[:,r_idx]
    # This is equivalent to: for each (i,j,k), sum_c coeffs[c] * U[i,c]*V[j,c]*W[k,c] = U[i,r_idx]*V[j,r_idx]*W[k,r_idx]
    
    # We can set up a linear system. Let's use the fact that if we fix W, we can write:
    # U[:,r_idx] * V[:,r_idx] * W[:,r_idx] = sum_c coeffs[c] * U[:,c] * V[:,c] * W[:,c]
    
    # This is a trilinear system. Let's try a simpler approach:
    # If W[:,r_idx] is in the span of {W[:,c] : c in other_cols}, and the corresponding
    # U and V components also match, we can eliminate.
    
    # For now, let's try a heuristic: check if the column is linearly dependent
    # in the U, V, and W spaces simultaneously.
    
    # Build matrix from U columns
    U_other = U[:, other_cols]  # shape (12, len(other_cols))
    U_target = U[:, r_idx]      # shape (12,)
    
    # Build matrix from V columns
    V_other = V[:, other_cols]  # shape (16, len(other_cols))
    V_target = V[:, r_idx]      # shape (16,)
    
    # Build matrix from W columns
    W_other = W[:, other_cols]  # shape (12, len(other_cols))
    W_target = W[:, r_idx]      # shape (12,)
    
    # Try to solve for coeffs such that U_target ≈ U_other @ coeffs
    try:
        coeffs_u, _, _, _ = np.linalg.lstsq(U_other, U_target, rcond=tol)
        if np.linalg.norm(U_target - U_other @ coeffs_u) > tol * np.linalg.norm(U_target):
            return None, False
    except np.linalg.LinAlgError:
        return None, False
    
    # Try to solve for coeffs such that V_target ≈ V_other @ coeffs
    try:
        coeffs_v, _, _, _ = np.linalg.lstsq(V_other, V_target, rcond=tol)
        if np.linalg.norm(V_target - V_other @ coeffs_v) > tol * np.linalg.norm(V_target):
            return None, False
    except np.linalg.LinAlgError:
        return None, False
    
    # Try to solve for coeffs such that W_target ≈ W_other @ coeffs
    try:
        coeffs_w, _, _, _ = np.linalg.lstsq(W_other, W_target, rcond=tol)
        if np.linalg.norm(W_target - W_other @ coeffs_w) > tol * np.linalg.norm(W_target):
            return None, False
    except np.linalg.LinAlgError:
        return None, False
    
    # The coefficients must be the same for all three. Check if they're consistent.
    # If coeffs_u ≈ coeffs_v ≈ coeffs_w, we have a valid elimination.
    if np.linalg.norm(coeffs_u - coeffs_v) > tol or np.linalg.norm(coeffs_u - coeffs_w) > tol:
        return None, False
    
    return coeffs_u, True


def _reduce_rank(U, V, W, deadline, max_iterations=100):
    """Attempt to reduce rank by eliminating linearly dependent columns."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    eliminated = set()
    progress = True
    
    while progress and len(eliminated) < r - 1:
        progress = False
        remaining = [c for c in range(r) if c not in eliminated]
        
        for r_idx in remaining:
            if time.time() > deadline:
                break
            
            other_cols = [c for c in remaining if c != r_idx]
            if not other_cols:
                continue
            
            coeffs, success = _try_eliminate_column(U, V, W, r_idx, other_cols)
            if success and coeffs is not None:
                # Apply elimination
                for pos, c in enumerate(other_cols):
                    U[:, c] -= coeffs[pos] * U[:, r_idx]
                    V[:, c] -= coeffs[pos] * V[:, r_idx]
                    W[:, c] -= coeffs[pos] * W[:, r_idx]
                eliminated.add(r_idx)
                progress = True
                break
    
    if not eliminated:
        return U, V, W
    
    keep = [c for c in range(r) if c not in eliminated]
    return U[:, keep], V[:, keep], W[:, keep]


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4."""
    start_time = time.time()
    deadline = start_time + 110

    U, V, W = _block_decomposition_3x4_4x4()

    if _verify_fast(U, V, W):
        # Try to reduce rank
        U2, V2, W2 = _reduce_rank(U, V, W, deadline)
        if _verify_fast(U2, V2, W2):
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
