"""Matrix multiplication 2x2x10: rank-minimal exact tensor decomposition.

Strategy: Construct the rank-35 Strassen baseline efficiently, then attempt
to find a lower-rank (30-34) decomposition using a robust alternating least
squares approach with multiple restarts and exact verification.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_DEADLINE = 120.0


def _build_strassen_baseline():
    """Build the safe rank-35 Strassen baseline."""
    n, m, p = 2, 2, 10
    num_blocks = p // 2  # 5 blocks
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7-multiplication decomposition for 2x2x2
    # Each entry: (U_indices, V_indices, W_indices)
    # U is 4x1 (2x2 matrix A), V is 4x1 (2x2 matrix B), W is 4x1 (2x2 matrix C)
    # Indices: 0=A00, 1=A01, 2=A10, 3=A11 for U
    #           0=B00, 1=B01, 2=B10, 3=B11 for V  
    #           0=C00, 1=C01, 2=C10, 3=C11 for W
    
    strassen_decomp = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    for b in range(num_blocks):
        base_col = b * 7
        for t in range(7):
            col = base_col + t
            u_side, v_side, w_side = strassen_decomp[t]
            for u_row, u_coeff in u_side:
                U[u_row][col] = u_coeff
            for v_koff, v_coeff in v_side:
                j = v_koff // 2
                k = 2 * b + v_koff % 2
                V[j * p + k][col] = v_coeff
            for w_koff, w_coeff in w_side:
                i = w_koff // 2
                k = 2 * b + w_koff % 2
                W[k * n + i][col] = w_coeff
    
    return U, V, W, rank


def _verify_decomposition(U, V, W, rank, n, m, p):
    """Verify that the decomposition exactly performs matrix multiplication."""
    rng = np.random.default_rng(12345)
    A_test = rng.integers(-5, 6, size=(n, m))
    B_test = rng.integers(-5, 6, size=(m, p))
    C_expected = A_test @ B_test
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
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
    
    return np.allclose(C_computed, C_expected_reorder, atol=1e-9)


def _try_als_decomposition(T, n, m, p, target_rank, max_iters=200, tol=1e-10):
    """Try to find a rank-`target_rank` decomposition using ALS."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # T shape: (nm, mp, pn)
    T_vec_u = T.reshape(nm, -1)  # (nm, mp*pn)
    T_vec_v = T.transpose(1, 0, 2).reshape(mp, -1)  # (mp, nm*pn)
    T_vec_w = T.transpose(2, 0, 1).reshape(pn, -1)  # (pn, nm*mp)
    
    best_err = np.inf
    best_U = None
    best_V = None
    best_W = None
    
    for restart in range(5):
        rng = np.random.default_rng(42 + restart * 1000 + target_rank)
        U = rng.normal(size=(nm, target_rank))
        V = rng.normal(size=(mp, target_rank))
        W = rng.normal(size=(pn, target_rank))
        
        for iteration in range(max_iters):
            # Update U: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For fixed i: T[i,:, :] = sum_r U[i,r] * (V[:,r] outer W[:,r])
            C_mat = np.einsum('jr,kr->rjk', V, W)  # (r, mp, pn)
            C_flat = C_mat.reshape(target_rank, -1)  # (r, mp*pn)
            
            try:
                # U[i,:] @ C_flat.T = T[i, :].flatten()
                # U = T_vec_u @ C_flat.T @ pinv(C_flat @ C_flat.T)
                A_mat = C_flat @ C_flat.T  # (r, r)
                U = T_vec_u @ C_flat.T @ np.linalg.pinv(A_mat)
            except:
                pass
            
            # Update V
            C_mat = np.einsum('ir,kr->rik', U, W)  # (r, nm, pn)
            C_flat = C_mat.reshape(target_rank, -1)  # (r, nm*pn)
            try:
                A_mat = C_flat @ C_flat.T
                V = T_vec_v @ C_flat.T @ np.linalg.pinv(A_mat)
            except:
                pass
            
            # Update W
            C_mat = np.einsum('ir,jr->rij', U, V)  # (r, nm, mp)
            C_flat = C_mat.reshape(target_rank, -1)  # (r, nm*mp)
            try:
                A_mat = C_flat @ C_flat.T
                W = T_vec_w @ C_flat.T @ np.linalg.pinv(A_mat)
            except:
                pass
            
            # Compute error
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T_recon - T)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < tol:
                break
        
        if best_err < tol:
            break
    
    if best_err < tol:
        return best_U, best_V, best_W, True
    
    return None, None, None, False


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with the safe baseline
    U, V, W, rank = _build_strassen_baseline()
    
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = 1
    nm = n * m  # 4
    mp = m * p  # 20
    pn = p * n  # 20
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find a lower-rank decomposition
    for target_rank in [30, 31, 32, 33, 34]:
        if time.time() - start_time > _DEADLINE - 30:
            break
        
        U_try, V_try, W_try, success = _try_als_decomposition(
            T, n, m, p, target_rank, max_iters=100, tol=1e-10
        )
        
        if success:
            # Verify with integer test
            if _verify_decomposition(U_try.tolist(), V_try.tolist(), W_try.tolist(),
                                     target_rank, n, m, p):
                rank = target_rank
                U = U_try.tolist()
                V = V_try.tolist()
                W = W_try.tolist()
                break
    
    # Final verification
    if not _verify_decomposition(U, V, W, rank, n, m, p):
        U, V, W, rank = _build_strassen_baseline()
    
    return {"U": U, "V": V, "W": W}
