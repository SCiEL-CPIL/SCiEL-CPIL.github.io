"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    """
    factors = []
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return factors

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using recursive Strassen.
    """
    n, m, p = DIMENSIONS
    s2 = _strassen_2x2_factors()

    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t in range(2):
        for s in range(7):
            for r in range(7):
                col = t * 49 + s * 7 + r

                u_block = s2[s][0]
                v_block = s2[s][1]
                w_block = s2[s][2]
                u_scalar = s2[r][0]
                v_scalar = s2[r][1]
                w_scalar = s2[r][2]

                for bi in range(2):
                    for bj in range(2):
                        cb = u_block[bi * 2 + bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = u_scalar[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                i = 2 * bi + ei
                                j = 2 * bj + ej
                                U[i * m + j][col] += cb * cs

                for bi in range(2):
                    for bj in range(2):
                        cb = v_block[bi * 2 + bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = v_scalar[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                j = 2 * bi + ei
                                k = 4 * t + 2 * bj + ej
                                V[j * p + k][col] += cb * cs

                for ci in range(2):
                    for cj in range(2):
                        cb = w_block[ci * 2 + cj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = w_scalar[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                i = 2 * ci + ei
                                k = 4 * t + 2 * cj + ej
                                W[k * n + i][col] += cb * cs

    return {"U": U, "V": V, "W": W}

def _compute_tensor(U_arr, V_arr, W_arr):
    """Compute the full tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]."""
    n, m, p = DIMENSIONS
    rank = U_arr.shape[1]
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return T

def _try_remove_column(U_arr, V_arr, W_arr, r, tol=1e-8):
    """
    Try to remove column r by checking if it's a linear combination of others.
    Uses efficient pre-computed column contributions.
    Returns (success, U_arr, V_arr, W_arr).
    """
    n, m, p = DIMENSIONS
    rank = U_arr.shape[1]
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # Target: contribution of column r
    target = U3[:, :, r] * V3[:, :, r] * W3[:, :, r]
    target_flat = target.flatten()
    
    # Build matrix of other columns' contributions
    mask = [s for s in range(rank) if s != r]
    other_T = np.stack([U3[:, :, s] * V3[:, :, s] * W3[:, :, s] for s in mask], axis=-1)
    other_T_flat = other_T.reshape(-1, len(mask))
    
    # Solve least squares
    try:
        c, residuals, rank_ls, sv = np.linalg.lstsq(other_T_flat, target_flat, rcond=None)
    except Exception:
        return False, U_arr, V_arr, W_arr
    
    residual = other_T_flat @ c - target_flat
    if np.linalg.norm(residual) < tol:
        # Remove column r
        U_new = np.delete(U_arr, r, axis=1)
        V_new = np.delete(V_arr, r, axis=1)
        W_new = np.delete(W_arr, r, axis=1)
        return True, U_new, V_new, W_new
    
    return False, U_arr, V_arr, W_arr

def _reduce_rank_linear(U, V, W, max_time=60.0):
    """
    Greedily remove columns that are linear combinations of the remaining columns.
    Optimized: pre-compute column contributions once per iteration.
    """
    start_time = time.time()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    n, m, p = DIMENSIONS
    tol = 1e-8
    
    while True:
        if time.time() - start_time > max_time:
            break
        rank = U_arr.shape[1]
        if rank <= 1:
            break
        
        # Pre-compute all column contributions
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        col_contribs = np.stack([U3[:, :, s] * V3[:, :, s] * W3[:, :, s] for s in range(rank)], axis=-1)
        col_contribs_flat = col_contribs.reshape(-1, rank)
        
        removed = False
        for r in range(rank):
            if time.time() - start_time > max_time:
                break
            
            # Target: contribution of column r
            target_flat = col_contribs_flat[:, r]
            
            # Build matrix of other columns' contributions
            mask = [s for s in range(rank) if s != r]
            other_T_flat = col_contribs_flat[:, mask]
            
            # Solve least squares
            try:
                c, residuals, rank_ls, sv = np.linalg.lstsq(other_T_flat, target_flat, rcond=None)
            except Exception:
                continue
            
            residual = other_T_flat @ c - target_flat
            if np.linalg.norm(residual) < tol:
                # Remove column r
                U_arr = np.delete(U_arr, r, axis=1)
                V_arr = np.delete(V_arr, r, axis=1)
                W_arr = np.delete(W_arr, r, axis=1)
                removed = True
                break
        
        if not removed:
            break
    
    return U_arr, V_arr, W_arr

def _polish_decomposition(U_arr, V_arr, W_arr, max_time=30.0):
    """
    Try to polish the decomposition to enable further rank reductions.
    Uses random linear combinations to find equivalent lower-rank decompositions.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = U_arr.shape[1]
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank
    
    # Try random linear combinations of columns
    np.random.seed(42)
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break
        
        # Generate random invertible matrix
        R = np.random.randn(rank, rank)
        try:
            det = np.linalg.det(R)
            if abs(det) < 1e-6:
                continue
        except Exception:
            continue
        
        # Apply transformation: new_factors = old_factors @ R
        U_pert = U_arr @ R
        V_pert = V_arr @ R
        W_pert = W_arr @ R
        
        # Verify it's still valid
        U3_pert = U_pert.reshape(n, m, rank)
        V3_pert = V_pert.reshape(m, p, rank)
        W3_pert = W_pert.reshape(p, n, rank)
        T_pert = np.einsum('ijr,jkr,kir->ijk', U3_pert, V3_pert, W3_pert)
        
        # Check against original
        U3_orig = U_arr.reshape(n, m, rank)
        V3_orig = V_arr.reshape(m, p, rank)
        W3_orig = W_arr.reshape(p, n, rank)
        T_orig = np.einsum('ijr,jkr,kir->ijk', U3_orig, V3_orig, W3_orig)
        
        if not np.allclose(T_pert, T_orig, atol=1e-6):
            continue
        
        # Try rank reduction on transformed version
        U_new, V_new, W_new = _reduce_rank_linear(
            U_pert.tolist(), V_pert.tolist(), W_pert.tolist(),
            max_time=3.0
        )
        
        new_rank = U_new.shape[1]
        if new_rank < best_rank:
            best_rank = new_rank
            best_U, best_V, best_W = U_new, V_new, W_new
    
    return best_U, best_V, best_W

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_rank = float('inf')
    
    try:
        result = _try_strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            best_result = result
            best_rank = len(result["U"][0])
            
            if time.time() < deadline - 20:
                U_arr = np.array(result["U"], dtype=float)
                V_arr = np.array(result["V"], dtype=float)
                W_arr = np.array(result["W"], dtype=float)
                
                # First pass: direct rank reduction (optimized)
                U_arr, V_arr, W_arr = _reduce_rank_linear(
                    U_arr, V_arr, W_arr,
                    max_time=25.0
                )
                new_rank = U_arr.shape[1]
                
                if _verify(U_arr.tolist(), V_arr.tolist(), W_arr.tolist()):
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_result = {
                            "U": U_arr.tolist(),
                            "V": V_arr.tolist(),
                            "W": W_arr.tolist(),
                        }
                
                # Second pass: polish with random transformations
                if time.time() < deadline - 15:
                    U_arr, V_arr, W_arr = _polish_decomposition(
                        U_arr, V_arr, W_arr,
                        max_time=25.0
                    )
                    new_rank = U_arr.shape[1]
                    
                    if _verify(U_arr.tolist(), V_arr.tolist(), W_arr.tolist()):
                        if new_rank < best_rank:
                            best_rank = new_rank
                            best_result = {
                                "U": U_arr.tolist(),
                                "V": V_arr.tolist(),
                                "W": W_arr.tolist(),
                            }
                
                # Final attempt: more aggressive reduction
                if time.time() < deadline - 5:
                    U_arr, V_arr, W_arr = _reduce_rank_linear(
                        U_arr, V_arr, W_arr,
                        max_time=8.0
                    )
                    new_rank = U_arr.shape[1]
                    
                    if _verify(U_arr.tolist(), V_arr.tolist(), W_arr.tolist()):
                        if new_rank < best_rank:
                            best_rank = new_rank
                            best_result = {
                                "U": U_arr.tolist(),
                                "V": V_arr.tolist(),
                                "W": W_arr.tolist(),
                            }
    except Exception as e:
        pass
    
    if best_result is not None:
        return best_result
    
    return _schoolbook()
# EVOLVE-BLOCK-END
