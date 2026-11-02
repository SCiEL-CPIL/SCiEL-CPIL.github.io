import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def _build_tensor3(n, m, p):
    """Build the 3-way multiplication tensor T[i,j,k] = 1 for all i,j,k
    (each entry represents one independent product term in the bilinear identity)."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _validate_decomp(U, V, W, n, m, p, tol=1e-8):
    """Validate that sum_r u_r[i]*v_r[j]*w_r[k] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reconstruct T3[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T3_recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T3_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    target = np.ones((n, m, p))
    return np.allclose(T3_recon, target, atol=tol)

def _peel_exact(T3, n, m, p, max_terms, seed):
    """Exact rank-1 peeling: at each step, pick a nonzero entry (i,j,k),
    extract the three slices, and subtract the rank-1 tensor.
    Returns (U, V, W) if successful, else None."""
    rng = np.random.RandomState(seed)
    T_rem = T3.copy()
    U_cols = []
    V_cols = []
    W_cols = []
    
    for r in range(max_terms):
        # Find all nonzero entries
        nz = np.nonzero(np.abs(T_rem) > 1e-10)
        if len(nz[0]) == 0:
            break
        
        # Pick a random nonzero entry
        idx = rng.randint(0, len(nz[0]))
        i, j, k = nz[0][idx], nz[1][idx], nz[2][idx]
        val = T_rem[i, j, k]
        
        # Extract slices
        u_slice = T_rem[:, j, k].copy()  # shape (n,)
        v_slice = T_rem[i, :, k].copy()  # shape (m,)
        w_slice = T_rem[i, j, :].copy()  # shape (p,)
        
        # The rank-1 tensor is u ⊗ v ⊗ w scaled appropriately
        # We want (u⊗v⊗w)[i,j,k] = u[i]*v[j]*w[k] = val
        # So scale = val / (u_slice[i]*v_slice[j]*w_slice[k])
        prod = u_slice[i] * v_slice[j] * w_slice[k]
        if abs(prod) < 1e-10:
            continue
        
        scale = val / prod
        u = u_slice * scale
        v = v_slice
        w = w_slice
        
        # Now u[i]*v[j]*w[k] = scale * u_slice[i] * v_slice[j] * w_slice[k] = val
        # Subtract this rank-1 tensor from T_rem
        T_rem -= np.einsum('i,j,k->ijk', u, v, w)
        
        # Build U, V, W columns
        # U has shape (n*m, rank): U[i*m+j, r] = u[i] for all j? No!
        # Wait - the rank-1 tensor is u[i] * v[j] * w[k]
        # So U[i*m+j, r] should be u[i] * (something that depends on j?)
        # Actually no: the tensor entry is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need this to equal u[i] * v[j] * w[k]
        # One valid factorization: U[i*m+j, r] = u[i], V[j*p+k, r] = v[j], W[k*n+i, r] = w[k]
        # But then U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = u[i]*v[j]*w[k] ✓
        
        U_col = np.zeros(n * m)
        V_col = np.zeros(m * p)
        W_col = np.zeros(p * n)
        
        for ii in range(n):
            for jj in range(m):
                U_col[ii * m + jj] = u[ii]
        
        for jj in range(m):
            for kk in range(p):
                V_col[jj * p + kk] = v[jj]
        
        for kk in range(p):
            for ii in range(n):
                W_col[kk * n + ii] = w[kk]
        
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    if len(U_cols) == 0:
        return None
    
    # Check if residual is zero
    if np.max(np.abs(T_rem)) > 1e-8:
        return None
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return U, V, W

def _column_space_lower_bound(n, m, p):
    """Compute the column-space rank lower bound for the multiplication tensor.
    The (n*m) x (p*n) unfolding has rank at most n*min(m,p) but the actual rank
    of the multiplication tensor's matrix flattening gives a lower bound."""
    # Build the (n*m) x (p*n) matrix where M[i*m+j, k*n+i] = 1 if valid
    # Actually, the standard lower bound is the rank of the "slice" matrix
    # For matmul(n,m,p), a lower bound is max(rank of A-slices, rank of B-slices, rank of C-slices)
    # The (n*m) x (p*n) matrix: row (i,j), col (k,i') -> 1 if i=i' and ... 
    # Let's just use the known result: for matmul, the tensor rank >= n*min(m,p) in general
    # But for our specific case, let's compute the matrix rank of a flattening
    M = np.zeros((n * m, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i * m + j, k * n + i] = 1.0
    r = np.linalg.matrix_rank(M, tol=1e-9)
    return int(r)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 20s safety margin
    
    # Schoolbook fallback: rank = n*m*p
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    T3 = _build_tensor3(n, m, p)
    
    # Get lower bound
    lb = _column_space_lower_bound(n, m, p)
    
    # Try ranks from lower bound up to schoolbook
    # For 4x5x5, lb is likely around 20-30, schoolbook is 100
    # Try a range of candidate ranks
    candidate_ranks = list(range(lb, min(lb + 50, rank_sb + 1)))
    
    for R in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        # Try multiple random peeling attempts
        found = False
        for attempt in range(50):
            if time.time() > deadline - 3:
                break
            
            seed = attempt * 1000 + R
            result = _peel_exact(T3, n, m, p, R, seed)
            
            if result is not None:
                U, V, W = result
                # Verify
                if _validate_decomp(U, V, W, n, m, p, tol=1e-8):
                    if R < best_rank:
                        best_rank = R
                        best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        # If we found a valid decomposition at rank R, we can stop
        # (since we're going from low to high, this is the minimum)
        if found:
            break
    
    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
