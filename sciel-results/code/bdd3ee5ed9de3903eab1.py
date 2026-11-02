import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-6):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    target = _build_tensor(n, m, p)
    return np.allclose(T, target, atol=tol)

def _find_low_rank_decomp(T, target_rank, max_iter=10000, seed=42):
    """Try to find a decomposition of T with at most target_rank terms.
    Uses iterative rank-1 peeling with randomization."""
    rng = np.random.RandomState(seed)
    T_rem = T.copy()
    U_cols = []
    V_cols = []
    W_cols = []
    
    for r in range(target_rank):
        # Find a rank-1 component to peel off
        # Strategy: randomly pick a nonzero entry, use the slices as initial guess
        nz = np.nonzero(np.abs(T_rem) > 1e-10)
        if len(nz[0]) == 0:
            break
        # Pick random nonzero entry
        idx = rng.randint(0, len(nz[0]))
        i, j, k = nz[0][idx], nz[1][idx], nz[2][idx]
        val = T_rem[i, j, k]
        
        # Get slices
        u = T_rem[:, j, k].copy()
        v = T_rem[i, :, k].copy()
        w = T_rem[i, j, :].copy()
        
        # We want u[i]*v[j]*w[k] = val
        # Scale: let u' = u/val, then u'[i]=1, and we need v[j]*w[k]=1
        # This is getting complicated. Let me use a simpler approach.
        
        # Just use the slices as is, and subtract
        # The rank-1 tensor is u ⊗ v ⊗ w, but it won't exactly match T_rem
        # This is a heuristic
        
        # Better: use the fact that if T_rem[i,j,k] = val, and we set
        # u = T_rem[:,j,k], v = T_rem[i,:,k], w = T_rem[i,j,:]
        # then (u⊗v⊗w)[i,j,k] = u[i]*v[j]*w[k] = T_rem[i,j,k]^3 / val^2 ... no
        
        # Let me just do a simple projection
        # The best rank-1 approximation in the direction of the slices
        # is u ⊗ v ⊗ w / (u[i]*v[j]*w[k]) * val if we want it to match at (i,j,k)
        
        u = T_rem[:, j, k].copy()
        v = T_rem[i, :, k].copy()
        w = T_rem[i, j, :].copy()
        
        prod = u[i] * v[j] * w[k]
        if abs(prod) < 1e-10:
            continue
        
        # Scale so that the product at (i,j,k) equals val
        scale = val / prod
        u = u * scale
        # Now u[i]*v[j]*w[k] = val
        
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
        T_rem -= np.einsum('i,j,k->ijk', u, v, w)
    
    if len(U_cols) == 0:
        return None, None, None, T_rem
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return U, V, W, T_rem

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    T = _build_tensor(n, m, p)
    
    # Try various target ranks
    for target_rank in [16, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        if time.time() > deadline - 10:
            break
        
        best_for_rank = None
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            U, V, W, T_rem = _find_low_rank_decomp(T, target_rank, seed=seed)
            if U is not None:
                if _validate(U, V, W, n, m, p):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U, V, W
                    break
                # Check if close
                if np.max(np.abs(T_rem)) < 1e-4:
                    # Close but not exact; try to fix
                    pass
    
    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
