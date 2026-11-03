import time
import numpy as np

DIMENSIONS = [2, 3, 4]
_T0 = time.time()
_DEADLINE = _T0 + 140.0  # 10s safety margin

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Check if U, V, W form a valid decomposition of the multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    if not U or not V or not W:
        return False
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The reconstruction is T_recon[(i,j), (j,k), (k,i)] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # We need this to be 1 for all i,j,k.
    # And 0 for any other combination of indices?
    # The tensor is defined on the product space.
    # The "support" of the multiplication tensor is the set of triples ((i,j), (j,k), (k,i))
    # where i in [0,n), j in [0,m), k in [0,p).
    # For any other combination of indices (a, b, c) in the flattened spaces,
    # the tensor entry is 0.
    # However, the indices in U, V, W are flattened.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We must check that the reconstructed tensor matches the target tensor.
    # Target tensor T has T[a, b, c] = 1 if a,b,c correspond to valid (i,j), (j,k), (k,i).
    # Otherwise 0.
    
    # Efficient check:
    # Instead of building the full 6x12x8 tensor, we can just check the non-zero entries
    # and a few random zero entries to ensure sparsity is maintained (though if the rank is low,
    # it's hard to accidentally hit a non-zero entry with 1.0 and miss it).
    # Actually, the condition is:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for any "invalid" combination of indices, the sum should be 0.
    # An invalid combination is a tuple (a, b, c) where a=i*m+j, b=j'*p+k', c=k''*n+i''
    # such that j != j' or k' != k'' or i'' != i.
    
    # Let's check the 24 valid entries first.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], np.dot(V[idx_v, :], W[idx_w, :]))
                # Note: np.dot(U[idx_u,:], W[idx_w,:]) is not right.
                # sum_c U[idx_u,c] V[idx_v,c] W[idx_w,c]
                # This is element-wise product then sum.
                prod = U[idx_u, :] * V[idx_v, :] * W[idx_w, :]
                if not np.isclose(np.sum(prod), 1.0, atol=tol):
                    return False
    
    # Check a sample of invalid entries to ensure they are 0.
    # If the rank is low, the factors are likely to produce non-zero values elsewhere.
    # We need to ensure they are 0.
    # Let's check all 576 entries to be safe, it's small.
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Determine if this is a valid entry
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                is_valid = (j == j2) and (k == k2) and (i == i2)
                
                prod = U[a, :] * V[b, :] * W[c, :]
                val = np.sum(prod)
                
                if is_valid:
                    if not np.isclose(val, 1.0, atol=tol):
                        return False
                else:
                    if not np.isclose(val, 0.0, atol=tol):
                        return False
                        
    return True

def _snap(x):
    """Snap near-integer / half-integer floats to exact rationals."""
    r = np.rint(2.0 * x) / 2.0
    return r

def _als_decompose(T, R, n_iter=50, seed=42):
    """
    Alternating Least Squares for tensor decomposition.
    T: shape (A, B, C)
    R: rank
    Returns U, V, W or None.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(seed)
    
    U = rng.randn(A, R) * 0.1
    V = rng.randn(B, R) * 0.1
    W = rng.randn(C, R) * 0.1
    
    for it in range(n_iter):
        # Solve for U
        # T[a, :, :] = sum_r U[a, r] * (V[:, r] * W[:, r])
        # Let M_VW[r, b, c] = V[b, r] * W[c, r]
        # T[a, b, c] = sum_r U[a, r] M_VW[r, b, c]
        # Flatten: T_flat[a, (b,c)] = sum_r U[a, r] M_VW_flat[r, (b,c)]
        
        # Construct matrix for U update
        # A_U is (B*C, R), B_U is (A, B*C)
        # We solve for each row a of U: U[a, :] = argmin sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
        
        # Precompute V and W outer products per rank
        # M_VW_flat shape (R, B*C)
        M_VW = np.zeros((R, B * C))
        for r in range(R):
            M_VW[r, :] = (V[:, r] * W[:, r]).flatten()
            
        # T_flat shape (A, B*C)
        T_flat = T.reshape(A, B * C)
        
        # Solve least squares: T_flat[a, :] = U[a, :] @ M_VW.T
        # U[a, :] = argmin || T_flat[a, :] - U[a, :] M_VW.T ||
        # This is a linear system: M_VW U[a, :].T = T_flat[a, :].T
        # U[a, :].T = lstsq(M_VW, T_flat[a, :].T)
        
        U_new = np.zeros_like(U)
        for a in range(A):
            U_new[a, :], _, _, _ = np.linalg.lstsq(M_VW, T_flat[a, :], rcond=None)
        U = U_new
        
        # Solve for V
        M_UW = np.zeros((R, A * C))
        for r in range(R):
            M_UW[r, :] = (U[:, r] * W[:, r]).flatten()
        T_flat_V = T.reshape(B, A * C) # T[b, a, c] -> need to permute?
        # T[a, b, c]. We want to fix b.
        # T[:, b, :] shape (A, C). Flatten to (A*C).
        # M_UW[r, a, c] = U[a, r] W[c, r].
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r] = V[b, r] * (U[:, r] * W[:, r])[a, c]
        # So T_flat_V[b, (a,c)] = sum_r V[b, r] M_UW[r, (a,c)]
        
        V_new = np.zeros_like(V)
        for b in range(B):
            # T_flat_V[b, :] is T[:, b, :].flatten()
            T_flat_V[b, :] = T[:, b, :].flatten()
            V_new[b, :], _, _, _ = np.linalg.lstsq(M_UW, T_flat_V[b, :], rcond=None)
        V = V_new
        
        # Solve for W
        M_UV = np.zeros((R, A * B))
        for r in range(R):
            M_UV[r, :] = (U[:, r] * V[:, r]).flatten()
        T_flat_W = np.zeros((C, A * B))
        for c in range(C):
            T_flat_W[c, :] = T[:, :, c].flatten()
            
        W_new = np.zeros_like(W)
        for c in range(C):
            W_new[c, :], _, _, _ = np.linalg.lstsq(M_UV, T_flat_W[c, :], rcond=None)
        W = W_new
        
        # Check convergence
        if it % 10 == 0:
            # Quick check
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                        err += (val - 1.0)**2
            if err < 1e-10:
                break
                
    # Final check
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                err += (val - 1.0)**2
    if err < 1e-8:
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try ranks lower than schoolbook
    # Start from rank 22, 23
    for r in [22, 23]:
        if time.time() > _DEADLINE - 10:
            break
            
        # Try multiple seeds
        # Use a few seeds to increase chance
        for seed in range(20):
            if time.time() > _DEADLINE - 2:
                break
                
            res = _als_decompose(T, r, n_iter=100, seed=seed)
            if res is not None:
                U, V, W = res
                # Snap to half-integers
                U_snap = _snap(U)
                V_snap = _snap(V)
                W_snap = _snap(W)
                
                # Verify
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                
                # Try integer snap
                U_int = np.rint(U)
                V_int = np.rint(V)
                W_int = np.rint(W)
                if _verify(U_int, V_int, W_int, n, m, p):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }

    # Fallback: Schoolbook
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
