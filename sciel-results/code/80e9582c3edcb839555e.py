import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 3, 6]

def _build_schoolbook(n, m, p):
    """Build the standard (schoolbook) matrix multiplication tensor decomposition."""
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    a_dim, b_dim, c_dim = n * m, m * p, p * n
    R = U.shape[1]
    
    # Compute the reconstructed tensor
    # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    # Build target tensor
    T_target = np.zeros((a_dim, b_dim, c_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    return np.allclose(T_recon, T_target, atol=tol)

def _try_greedy_removal(U, V, W, n, m, p, deadline):
    """
    Greedily try to remove columns one at a time if the remaining columns
    can still exactly reconstruct the tensor.
    
    For each column r, check if there exist vectors u', v', w' such that
    sum_{r' != r} U[:,r']V[:,r']W[:,r'] + u'v'w' = T
    This is equivalent to checking if T - sum_{r' != r} ... is rank 1.
    """
    R = U.shape[1]
    a_dim, b_dim, c_dim = U.shape[0], V.shape[0], W.shape[0]
    
    # Precompute target tensor
    T_target = np.zeros((a_dim, b_dim, c_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    active = list(range(R))
    
    # Repeatedly try to remove a column
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r_idx in list(active):
            if len(active) <= 1:
                break
            
            # Compute sum of all other columns
            # S[a,b,c] = sum_{r' != r_idx} U[a,r']V[b,r']W[c,r']
            # = Total - Col_r_idx
            
            Total = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            Col_r = np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx], optimize=True)
            S = Total - Col_r
            
            # Residual = T - S, check if rank 1
            Residual = T_target - S
            
            # Check if Residual is rank 1
            # A tensor is rank 1 if all its slices are proportional
            # Find a non-zero entry
            nonzero = np.argwhere(np.abs(Residual) > 1e-10)
            if len(nonzero) == 0:
                # Residual is zero, so we can just remove this column
                active.remove(r_idx)
                changed = True
                continue
            
            a0, b0, c0 = nonzero[0]
            val = Residual[a0, b0, c0]
            
            if abs(val) < 1e-10:
                continue
            
            # If rank 1: Residual[a,b,c] = u[a]*v[b]*w[c]
            # Then: u[a] = Residual[a,b0,c0]/val, v[b] = Residual[a0,b,c0]/val, w[c] = Residual[a0,b0,c]/val
            u_vec = Residual[:, b0, c0] / val
            v_vec = Residual[a0, :, c0] / val
            w_vec = Residual[a0, b0, :] / val
            
            # Verify
            recon = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec, optimize=True)
            if np.allclose(recon, Residual, atol=1e-8):
                # Update this column to the new rank-1 term
                U[:, r_idx] = u_vec
                V[:, r_idx] = v_vec
                W[:, r_idx] = w_vec
                
                # Check if this column is now zero (can be removed)
                if (np.allclose(U[:, r_idx], 0, atol=1e-10) and 
                    np.allclose(V[:, r_idx], 0, atol=1e-10) and 
                    np.allclose(W[:, r_idx], 0, atol=1e-10)):
                    active.remove(r_idx)
                    changed = True
                else:
                    # Keep the column but with updated values
                    pass
    
    # Verify final result
    if not active:
        return U, V, W, 0
    
    U_final = U[:, active]
    V_final = V[:, active]
    W_final = W[:, active]
    
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        return U_final, V_final, W_final, len(active)
    
    return None, None, None, None

def _try_pairwise_replacement(U, V, W, n, m, p, deadline):
    """
    Try to replace pairs of columns with a single column.
    For columns r1, r2, check if there exist u', v', w' such that
    sum_{r != r1,r2} ... + u'v'w' = T
    i.e., T - sum_{r != r1,r2} ... - Col_r1 - Col_r2 is rank 1.
    """
    R = U.shape[1]
    a_dim, b_dim, c_dim = U.shape[0], V.shape[0], W.shape[0]
    
    T_target = np.zeros((a_dim, b_dim, c_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    active = list(range(R))
    
    # Try to remove pairs
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r1_idx in range(len(active)):
            for r2_idx in range(r1_idx + 1, len(active)):
                r1 = active[r1_idx]
                r2 = active[r2_idx]
                
                # Compute sum of all other columns
                Total = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                Col_r1 = np.einsum('a,b,c->abc', U[:, r1], V[:, r1], W[:, r1], optimize=True)
                Col_r2 = np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2], optimize=True)
                S = Total - Col_r1 - Col_r2
                
                Residual = T_target - S
                
                # Check if rank 1
                nonzero = np.argwhere(np.abs(Residual) > 1e-10)
                if len(nonzero) == 0:
                    # Can remove both columns
                    active.remove(r2)
                    active.remove(r1)
                    improved = True
                    break
                else:
                    a0, b0, c0 = nonzero[0]
                    val = Residual[a0, b0, c0]
                    if abs(val) < 1e-10:
                        continue
                    
                    u_vec = Residual[:, b0, c0] / val
                    v_vec = Residual[a0, :, c0] / val
                    w_vec = Residual[a0, b0, :] / val
                    
                    recon = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec, optimize=True)
                    if np.allclose(recon, Residual, atol=1e-8):
                        # Replace r1 with the new term, remove r2
                        U[:, r1] = u_vec
                        V[:, r1] = v_vec
                        W[:, r1] = w_vec
                        active.remove(r2)
                        improved = True
                        break
            if improved:
                break
    
    if not active:
        return U, V, W, 0
    
    U_final = U[:, active]
    V_final = V[:, active]
    W_final = W[:, active]
    
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        return U_final, V_final, W_final, len(active)
    
    return None, None, None, None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120 - 15  # Safety margin
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy 1: Greedy single-column removal/replacement
    if time.time() < deadline - 30:
        U1, V1, W1, rank1 = _try_greedy_removal(U.copy(), V.copy(), W.copy(), n, m, p, deadline - 30)
        if U1 is not None and rank1 < best_rank:
            best_U, best_V, best_W = U1, V1, W1
            best_rank = rank1
    
    # Strategy 2: Pairwise replacement from current best
    if time.time() < deadline - 10:
        U2, V2, W2, rank2 = _try_pairwise_replacement(best_U.copy(), best_V.copy(), best_W.copy(), n, m, p, deadline - 10)
        if U2 is not None and rank2 < best_rank:
            best_U, best_V, best_W = U2, V2, W2
            best_rank = rank2
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook if something went wrong
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
