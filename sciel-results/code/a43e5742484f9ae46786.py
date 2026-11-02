"""Matrix multiplication 2x2x10: rank-minimal exact tensor decomposition.

Strategy: Start from the rank-35 block-diagonal Strassen decomposition (5 independent
2x2x2 blocks). Then attempt to reduce rank by:
1. Numerically searching for a rank-34 decomposition via alternating optimization
   with symmetry-adapted initialization.
2. If found, round to half-integers and verify.
3. Fall back to rank-35 if no valid rank-34 is found within time budget.

The key insight: the 2x2x10 tensor has invariance under column permutations of B.
The block-diagonal Strassen representation may obscure linear dependencies among
the 35 rank-1 terms. A global optimization in a symmetry-adapted basis can
potentially find a rank-34 decomposition if one exists.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_START_TIME = time.time()
_DEADLINE = 115.0  # safety margin before 120s kill


def _elapsed():
    return time.time() - _START_TIME


def _verify_decomp(U, V, W, rank, n, m, p, num_trials=8):
    """Verify decomposition with multiple random integer trials."""
    try:
        rng = np.random.default_rng(42)
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for trial in range(num_trials):
            A_test = rng.integers(-4, 5, size=(n, m))
            B_test = rng.integers(-4, 5, size=(m, p))
            C_expected = A_test @ B_test
            
            A_flat = A_test.flatten()
            B_flat = B_test.flatten()
            
            C_computed = np.zeros(p * n)
            for r in range(rank):
                uA = np.dot(U_arr[:, r], A_flat)
                vB = np.dot(V_arr[:, r], B_flat)
                C_computed += uA * vB * W_arr[:, r]
            
            C_expected_reorder = np.zeros(p * n)
            for i in range(n):
                for k in range(p):
                    C_expected_reorder[k * n + i] = C_expected[i, k]
            
            if not np.allclose(C_computed, C_expected_reorder, atol=1e-9):
                return False
        return True
    except Exception:
        return False


def _build_strassen_block(col_offset, U, V, W, n, m, p, block_col_start):
    """Add a 7-rank Strassen decomposition for a 2x2x2 block."""
    strassen_decomp = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    for t in range(7):
        col = col_offset + t
        u_side, v_side, w_side = strassen_decomp[t]
        for u_row, u_coeff in u_side:
            U[u_row][col] = u_coeff
        for v_koff, v_coeff in v_side:
            j = v_koff // 2
            k = block_col_start + v_koff % 2
            V[j * p + k][col] = v_coeff
        for w_koff, w_coeff in w_side:
            i = w_koff // 2
            k = block_col_start + w_koff % 2
            W[k * n + i][col] = w_coeff


def _build_rank35():
    """Build the baseline rank-35 Strassen decomposition."""
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        _build_strassen_block(b * 7, U, V, W, n, m, p, b * 2)
    
    return U, V, W, rank


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k]."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T


def _tensor_error(U, V, W, T):
    """Compute the Frobenius norm error of the tensor decomposition."""
    n, m, p = T.shape[0] // m, m, T.shape[2] // n
    nm, mp, pn = T.shape
    err = 0.0
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                computed = np.dot(U[i, :], V[j, :] * W[k, :])
                err += (T[i, j, k] - computed) ** 2
    return np.sqrt(err)


def _try_rank34_search(T, U35, V35, W35, time_budget=90.0):
    """
    Try to find a rank-34 decomposition by starting from the rank-35 Strassen
    decomposition and using alternating optimization with a penalty for
    non-zero entries in the last column, effectively pushing toward rank 34.
    
    Strategy: Initialize with the rank-35 solution, then optimize a rank-34
    sub-decomposition by randomly zeroing one column and re-optimizing.
    """
    n, m, p = 2, 2, 10
    nm, mp, pn = n * m, m * p, p * n  # 4, 20, 20
    
    U_init = np.array(U35, dtype=float)
    V_init = np.array(V35, dtype=float)
    W_init = np.array(W35, dtype=float)
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try zeroing each of the 35 columns and see if the remaining 34 can
    # be re-optimized to recover the tensor.
    for zero_col in range(35):
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Remove column zero_col
        mask = [c for c in range(35) if c != zero_col]
        U0 = U_init[:, mask].copy()
        V0 = V_init[:, mask].copy()
        W0 = W_init[:, mask].copy()
        
        # Alternating optimization for rank-34
        U_cur, V_cur, W_cur = U0.copy(), V0.copy(), W0.copy()
        
        for it in range(150):
            if _elapsed() > _DEADLINE - 10:
                break
            
            # Update U: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For fixed V, W: T_flat[i, j*pn+k] = sum_r U[i,r] * (V[j,r]*W[k,r])
            # Build M: M[r, j*pn+k] = V[j,r]*W[k,r]
            M = np.einsum('jr,kr->rjk', V_cur, W_cur).reshape(34, mp * pn)
            T_flat = T.reshape(nm, mp * pn)
            
            try:
                MT_M = M.T @ M
                MT_T = M.T @ T_flat.T
                U_new = np.linalg.solve(MT_M + 1e-10 * np.eye(34), MT_T).T
            except np.linalg.LinAlgError:
                break
            
            # Update V: T[i,j,k] = sum_r V[j,r] * (U[i,r]*W[k,r])
            # For fixed U, W: T_flat2[j, i*pn+k] = sum_r V[j,r] * (U[i,r]*W[k,r])
            P = np.einsum('ir,kr->rik', U_new, W_cur).reshape(34, nm * pn)
            T_flat2 = T.reshape(mp, nm * pn)
            
            try:
                PT_P = P.T @ P
                PT_T = P.T @ T_flat2.T
                V_new = np.linalg.solve(PT_P + 1e-10 * np.eye(34), PT_T).T
            except np.linalg.LinAlgError:
                break
            
            # Update W: T[i,j,k] = sum_r W[k,r] * (U[i,r]*V[j,r])
            # For fixed U, V: T_flat3[k, i*mp+j] = sum_r W[k,r] * (U[i,r]*V[j,r])
            Q = np.einsum('ir,jr->rij', U_new, V_new).reshape(34, nm * mp)
            T_flat3 = T.reshape(pn, nm * mp)
            
            try:
                QT_Q = Q.T @ Q
                QT_T = Q.T @ T_flat3.T
                W_new = np.linalg.solve(QT_Q + 1e-10 * np.eye(34), QT_T).T
            except np.linalg.LinAlgError:
                break
            
            U_cur, V_cur, W_cur = U_new, V_new, W_new
            
            # Compute error
            err = _tensor_error(U_cur, V_cur, W_cur, T)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
            
            if err < 1e-12:
                break
        
        if best_err < 1e-10:
            break
    
    if best_err > 1e-8:
        return None
    
    # Round to nearest half-integer
    U_rounded = np.round(best_U * 2) / 2
    V_rounded = np.round(best_V * 2) / 2
    W_rounded = np.round(best_W * 2) / 2
    
    err = _tensor_error(U_rounded, V_rounded, W_rounded, T)
    if err > 1e-9:
        # Try integer rounding
        U_rounded = np.round(best_U)
        V_rounded = np.round(best_V)
        W_rounded = np.round(best_W)
        err = _tensor_error(U_rounded, V_rounded, W_rounded, T)
        if err > 1e-9:
            return None
    
    return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())


def _try_rank34_symmetry(T, time_budget=80.0):
    """
    Try to find a rank-34 decomposition using symmetry-adapted initialization.
    
    The 2x2x10 tensor is invariant under permutations of the 10 columns of B.
    We use this to create a more symmetric starting point for the optimization.
    
    Approach: Use the rank-35 Strassen solution, then apply a random
    orthogonal transformation to the 35 rank-1 terms, and try to find a
    rank-34 sub-decomposition.
    """
    n, m, p = 2, 2, 10
    nm, mp, pn = n * m, m * p, p * n
    
    U35, V35, W35, rank35 = _build_rank35()
    U_init = np.array(U35, dtype=float)
    V_init = np.array(V35, dtype=float)
    W_init = np.array(W35, dtype=float)
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try several random projections from rank 35 to rank 34
    for trial in range(20):
        if _elapsed() > _DEADLINE - 10:
            break
        
        rng = np.random.default_rng(1000 + trial)
        
        # Random rank-34 subspace of the rank-35 space
        # Generate a random 35x34 matrix and use it as projection
        R = rng.normal(size=(35, 34))
        Q, _ = np.linalg.qr(R)  # 35x34 orthonormal columns
        
        # Project the initial solution onto this subspace
        U0 = U_init @ Q  # 4x34
        V0 = V_init @ Q  # 20x34
        W0 = W_init @ Q  # 20x34
        
        # Alternating optimization
        U_cur, V_cur, W_cur = U0.copy(), V0.copy(), W0.copy()
        
        for it in range(100):
            if _elapsed() > _DEADLINE - 10:
                break
            
            # Update U
            M = np.einsum('jr,kr->rjk', V_cur, W_cur).reshape(34, mp * pn)
            T_flat = T.reshape(nm, mp * pn)
            try:
                MT_M = M.T @ M
                MT_T = M.T @ T_flat.T
                U_new = np.linalg.solve(MT_M + 1e-10 * np.eye(34), MT_T).T
            except np.linalg.LinAlgError:
                break
            
            # Update V
            P = np.einsum('ir,kr->rik', U_new, W_cur).reshape(34, nm * pn)
            T_flat2 = T.reshape(mp, nm * pn)
            try:
                PT_P = P.T @ P
                PT_T = P.T @ T_flat2.T
                V_new = np.linalg.solve(PT_P + 1e-10 * np.eye(34), PT_T).T
            except np.linalg.LinAlgError:
                break
            
            # Update W
            Qm = np.einsum('ir,jr->rij', U_new, V_new).reshape(34, nm * mp)
            T_flat3 = T.reshape(pn, nm * mp)
            try:
                QT_Q = Qm.T @ Qm
                QT_T = Qm.T @ T_flat3.T
                W_new = np.linalg.solve(QT_Q + 1e-10 * np.eye(34), QT_T).T
            except np.linalg.LinAlgError:
                break
            
            U_cur, V_cur, W_cur = U_new, V_new, W_new
            
            err = _tensor_error(U_cur, V_cur, W_cur, T)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
            
            if err < 1e-12:
                break
        
        if best_err < 1e-10:
            break
    
    if best_err > 1e-8:
        return None
    
    # Round to nearest half-integer
    U_rounded = np.round(best_U * 2) / 2
    V_rounded = np.round(best_V * 2) / 2
    W_rounded = np.round(best_W * 2) / 2
    
    err = _tensor_error(U_rounded, V_rounded, W_rounded, T)
    if err > 1e-9:
        U_rounded = np.round(best_U)
        V_rounded = np.round(best_V)
        W_rounded = np.round(best_W)
        err = _tensor_error(U_rounded, V_rounded, W_rounded, T)
        if err > 1e-9:
            return None
    
    return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())


def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    
    # Build rank-35 baseline (always works)
    U35, V35, W35, rank35 = _build_rank35()
    
    # Verify rank-35 baseline
    if not _verify_decomp(U35, V35, W35, rank35, n, m, p, num_trials=5):
        return {"U": U35, "V": V35, "W": W35}
    
    # Try to find rank-34 decomposition
    T = _build_tensor(n, m, p)
    
    # Method 1: Zero one column and re-optimize
    result34 = _try_rank34_search(T, U35, V35, W35)
    
    if result34 is None and _elapsed() < _DEADLINE - 15:
        # Method 2: Symmetry-adapted random projection
        result34 = _try_rank34_symmetry(T)
    
    if result34 is not None:
        U34, V34, W34 = result34
        if _verify_decomp(U34, V34, W34, 34, n, m, p, num_trials=8):
            return {"U": U34, "V": V34, "W": W34}
    
    # Fall back to rank-35
    return {"U": U35, "V": V35, "W": W35}
