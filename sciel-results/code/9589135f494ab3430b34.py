import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _build_strassen_decomp():
    """Build the rank-55 Strassen-based decomposition as numpy arrays."""
    n, m, p = DIMENSIONS
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    W_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    for block in range(5):
        k_offset = block * 2
        for s in range(7):
            r = col
            for local_idx, val in L[s].items():
                i = local_idx // 2
                j = local_idx % 2
                U[i * m + j, r] = val
            for local_idx, val in R[s].items():
                j = local_idx // 2
                k = local_idx % 2
                V[j * p + (k_offset + k), r] = val
            for local_idx, val in W_coeffs[s].items():
                i = local_idx // 2
                k = local_idx % 2
                W[(k_offset + k) * n + i, r] = val
            col += 1
    
    for i in range(2):
        for k in range(10):
            r = col
            U[i * m + 2, r] = 1
            V[2 * p + k, r] = 1
            W[k * n + i, r] = 1
            col += 1
    
    return U, V, W

def _tensor_to_matrix(U, V, W):
    """Convert factor matrices to the tensor in matrix form (p*n) x (n*m) x (m*p) flattened."""
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to verify: for all i,j,k, T[i,j,k] = delta(i,j,k) where the tensor is 1 iff the multiplication is correct.
    # Actually the tensor for matmul is: T[i,j,k] = 1 if (i,j,k) corresponds to a valid triple.
    # The matmul tensor: C_ik = sum_j A_ij B_jk
    # So T[i,j,k] = 1 for all valid (i,j,k) in the matmul tensor.
    # The decomposition satisfies: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = T[i,j,k]
    # where W is indexed as W[k*n+i, r] which is W[i,k,r] in (i,k) ordering.
    
    # Let's build the tensor as a 3D array
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1  # The matmul tensor has all 1s
    
    # Verify decomposition
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal T[i,j,k]
    return T

def _check_decomposition(U, V, W):
    """Check if the decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal 1 for all (i,j,k)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :])
                # Wait, that's not right. The tensor decomposition is:
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This is a trilinear form, not a bilinear one.
                # Let me compute it properly.
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _check_decomposition_fast(U, V, W):
    """Fast check using einsum."""
    n, m, p = DIMENSIONS
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape: U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # U[i*m+j, r] -> reshape to (n, m, r)
    # V[j*p+k, r] -> reshape to (m, p, r)
    # W[k*n+i, r] -> W is indexed as W[k*n+i], so reshape to (p, n, r)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    max_err = np.max(np.abs(T - 1.0))
    return max_err

def _als_decomposition(U, V, W, target_rank, max_iters=200, tol=1e-10, deadline=None):
    """ALS to find a lower-rank decomposition."""
    n, m, p = DIMENSIONS
    # Build the tensor
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Initialize with random or truncated SVD
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    for init in range(3):
        # Random initialization
        rng = np.random.RandomState(42 + init)
        Ur = rng.randn(n * m, target_rank) * 0.1
        Vr = rng.randn(m * p, target_rank) * 0.1
        Wr = rng.randn(p * n, target_rank) * 0.1
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U: for each (i,j), U[i*m+j, :] = argmin sum_k (T[i,j,k] - sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r])^2
            # This is a linear least squares problem in U[i*m+j, :]
            # Let's define: for fixed (i,j), we want to find u such that
            # sum_r u_r * (V[j*p+k,r] * W[k*n+i,r]) = T[i,j,k] for all k
            # This is: u @ M[i,j,:] = T[i,j,:] where M[i,j,k] = sum_r V[j*p+k,r] * W[k*n+i,r]
            
            # Build M matrix: M[i,j,k] = sum_r V[j*p+k,r] * W[k*n+i,r]
            # V[j*p+k, r] and W[k*n+i, r]
            # For each (i,j), M_k = sum_r V[j*p+k, r] * W[k*n+i, r]
            # This is: for each k, M_k = dot(V[j*p+k, :], W[k*n+i, :])
            
            # Efficient: M[i,j,k] = sum_r V[j*p+k,r] * W[k*n+i,r]
            # = sum_r V3[j,k,r] * W3[k,i,r]
            # = einsum('jkr,kir->ijk', V3, W3)
            M = np.einsum('jkr,kir->ijk', Vr.reshape(m, p, target_rank), Wr.reshape(p, n, target_rank))
            
            # Now for each (i,j), solve: U[i*m+j, :] @ M[i,j,:]^T = T[i,j,:]
            # i.e., U[i*m+j, :] = T[i,j,:] @ pinv(M[i,j,:].T)
            # Or: U[i*m+j, :] = lstsq(M[i,j,:].T, T[i,j,:])
            
            U_new = np.zeros((n * m, target_rank))
            for i in range(n):
                for j in range(m):
                    # M[i,j,:] is a vector of length p
                    # We want u such that u @ M[i,j,:] = T[i,j,:]
                    # This is: M[i,j,:]^T @ u = T[i,j,:]  (if we think of u as column)
                    # Actually: sum_r u_r * M[i,j,k] = T[i,j,k] for all k
                    # So: M[i,j,:]^T @ u = T[i,j,:]
                    # u = pinv(M[i,j,:].T) @ T[i,j,:]
                    Mij = M[i, j, :]  # shape (p,)
                    Tij = T[i, j, :]  # shape (p,)
                    # Solve Mij^T @ u = Tij
                    # This is a system of p equations in target_rank unknowns
                    # Use least squares
                    try:
                        u, _, _, _ = np.linalg.lstsq(Mij, Tij, rcond=None)
                    except:
                        u = np.zeros(target_rank)
                    U_new[i * m + j, :] = u
            
            # Update V: for each (j,k), V[j*p+k, :] = argmin
            # sum_r u_r * V[j*p+k,r] * W[k*n+i,r] = T[i,j,k] for all i
            # Let v = V[j*p+k, :], then for all i: sum_r v_r * (U[i*m+j,r] * W[k*n+i,r]) = T[i,j,k]
            # So: N[j,k,i] = sum_r U[i*m+j,r] * W[k*n+i,r]
            # v @ N[j,k,:] = T[:,j,k]
            
            N = np.einsum('ijr,kir->jki', Ur.reshape(n, m, target_rank), Wr.reshape(p, n, target_rank))
            # N[j,k,i] = sum_r U[i*m+j,r] * W[k*n+i,r]
            
            V_new = np.zeros((m * p, target_rank))
            for j in range(m):
                for k in range(p):
                    Njk = N[j, k, :]  # shape (n,)
                    Tjk = T[:, j, k]  # shape (n,)
                    try:
                        v, _, _, _ = np.linalg.lstsq(Njk, Tjk, rcond=None)
                    except:
                        v = np.zeros(target_rank)
                    V_new[j * p + k, :] = v
            
            # Update W: for each (k,i), W[k*n+i, :] = argmin
            # sum_r U[i*m+j,r] * V[j*p+k,r] * w_r = T[i,j,k] for all j
            # Let w = W[k*n+i, :], then for all j: sum_r w_r * (U[i*m+j,r] * V[j*p+k,r]) = T[i,j,k]
            # So: P[k,i,j] = sum_r U[i*m+j,r] * V[j*p+k,r]
            # w @ P[k,i,:] = T[i,:,k]
            
            P = np.einsum('ijr,jkr->kij', Ur.reshape(n, m, target_rank), Vr.reshape(m, p, target_rank))
            # P[k,i,j] = sum_r U[i*m+j,r] * V[j*p+k,r]
            
            W_new = np.zeros((p * n, target_rank))
            for k in range(p):
                for i in range(n):
                    Pki = P[k, i, :]  # shape (m,)
                    Tki = T[i, :, k]  # shape (m,)
                    try:
                        w, _, _, _ = np.linalg.lstsq(Pki, Tki, rcond=None)
                    except:
                        w = np.zeros(target_rank)
                    W_new[k * n + i, :] = w
            
            Ur, Vr, Wr = U_new, V_new, W_new
            
            # Check error
            U3 = Ur.reshape(n, m, target_rank)
            V3 = Vr.reshape(m, p, target_rank)
            W3 = Wr.reshape(p, n, target_rank)
            T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            err = np.max(np.abs(T_recon - T))
            
            if err < best_err:
                best_err = err
                best_U = Ur.copy()
                best_V = Vr.copy()
                best_W = Wr.copy()
            
            if err < tol:
                break
        
        if deadline is not None and time.time() > deadline:
            break
    
    return best_U, best_V, best_W, best_err

def _round_to_halfint(arr):
    """Round to nearest half-integer and check if it still works."""
    rounded = np.round(arr * 2) / 2
    return rounded

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the initial rank-55 decomposition
    U, V, W = _build_strassen_decomp()
    
    # Verify it's correct
    err = _check_decomposition_fast(U, V, W)
    if err > 1e-6:
        # Fallback: return the original
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 55
    best_result = (U.copy(), V.copy(), W.copy())
    
    # Try to reduce rank using ALS
    # Theoretical lower bound: max(n*m, m*p, p*n) / something... 
    # For matmul 2x3x10, the tensor has 60 non-zero entries.
    # Rank must be at least max(n*m, m*p, p*n) = max(6, 30, 20) = 30? No, that's not right.
    # The rank of the matmul tensor M_{n,m,p} is at least max(n*m, m*p, p*n) only in some cases.
    # Actually, the lower bound is more subtle. Let's try ranks from 30 to 54.
    
    # Try decreasing ranks
    for target_rank in range(54, 29, -1):
        if time.time() > deadline:
            break
        
        Ur, Vr, Wr, err = _als_decomposition(U, V, W, target_rank, max_iters=100, tol=1e-10, deadline=deadline)
        
        if Ur is not None and err < 1e-8:
            # Try to round to half-integers
            Ur_r = _round_to_halfint(Ur)
            Vr_r = _round_to_halfint(Vr)
            Wr_r = _round_to_halfint(Wr)
            
            # Check if rounded version still works
            err_rounded = _check_decomposition_fast(Ur_r, Vr_r, Wr_r)
            if err_rounded < 1e-6:
                best_rank = target_rank
                best_result = (Ur_r, Vr_r, Wr_r)
            else:
                # Use unrounded (but check if entries are close to half-integers)
                best_rank = target_rank
                best_result = (Ur, Vr, Wr)
            break  # Found a valid decomposition, stop
    
    U_best, V_best, W_best = best_result
    
    # Convert to lists
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
