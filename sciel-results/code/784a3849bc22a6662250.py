"""Tensor rank search for 2x3x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Schoolbook decomposition: rank = n*m*p = 54."""
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

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dense array.
    T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i (consistent indices), else 0.
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _tensor_error(U, V, W, T):
    """Compute the Frobenius norm of the tensor reconstruction error.
    T_reconstructed[a,b,c] = sum_k U[a,k] * V[b,k] * W[c,k]
    """
    # Efficient computation using einsum
    # T_rec = sum_k U[a,k] V[b,k] W[c,k]
    # This is a 3-way contraction
    T_rec = np.einsum('ak,bk,ck->abc', U, V, W)
    return np.linalg.norm(T_rec - T)

def _als_step_U(U, V, W, T, n, m, p):
    """ALS update for U. For each row a of U, solve a least squares problem."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    U_new = np.zeros_like(U)
    
    # Precompute V and W products for efficiency
    # For fixed a, we need to minimize sum_{b,c} (sum_k U[a,k] V[b,k] W[c,k] - T[a,b,c])^2
    # This is equivalent to: U[a,:] = (sum_{b,c} T[a,b,c] * (V[b,:] * W[c,:])) / (sum_{b,c} (V[b,:] * W[c,:])^2)
    # But this isn't quite right because of the cross terms.
    
    # Actually, let's do it properly. For fixed a, define:
    # M[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
    # We want to minimize sum_{b,c} (M[a,b,c] - T[a,b,c])^2 over U[a,:].
    
    # Let's use the fact that for fixed a, the problem is:
    # minimize ||U[a,:] * (V ⊙ W) - T[a,:,:]||^2
    # where (V ⊙ W) is a tensor of shape (m*p, p*n, r) with (V ⊙ W)[b,c,k] = V[b,k]*W[c,k]
    
    # This is a linear least squares problem:
    # A * x = b where A is (m*p*p*n) x r and b is (m*p*p*n)
    # A[flattened_bc, k] = V[b,k] * W[c,k]
    # b[flattened_bc] = T[a,b,c]
    
    # For efficiency, we can compute this without explicitly forming A:
    # A^T A = sum_{b,c} outer(V[b,:]*W[c,:], V[b,:]*W[c,:])
    # A^T b = sum_{b,c} T[a,b,c] * (V[b,:]*W[c,:])
    
    for a in range(n*m):
        # Compute A^T A and A^T b
        ATA = np.zeros((r, r), dtype=np.float64)
        ATb = np.zeros(r, dtype=np.float64)
        
        for b in range(m*p):
            for c in range(p*n):
                vw = V[b, :] * W[c, :]  # shape (r,)
                ATA += np.outer(vw, vw)
                if T[a, b, c] != 0:
                    ATb += T[a, b, c] * vw
        
        # Solve (ATA + lambda*I) x = ATb for regularization
        lam = 1e-8
        ATA_reg = ATA + lam * np.eye(r)
        try:
            U_new[a, :] = np.linalg.solve(ATA_reg, ATb)
        except np.linalg.LinAlgError:
            U_new[a, :] = np.linalg.lstsq(ATA_reg, ATb, rcond=None)[0]
    
    return U_new

def _als_step_V(U, V, W, T, n, m, p):
    """ALS update for V."""
    n, m, p = DIMENSIONS
    r = V.shape[1]
    V_new = np.zeros_like(V)
    
    for b in range(m*p):
        ATA = np.zeros((r, r), dtype=np.float64)
        ATb = np.zeros(r, dtype=np.float64)
        
        for a in range(n*m):
            for c in range(p*n):
                uw = U[a, :] * W[c, :]  # shape (r,)
                ATA += np.outer(uw, uw)
                if T[a, b, c] != 0:
                    ATb += T[a, b, c] * uw
        
        lam = 1e-8
        ATA_reg = ATA + lam * np.eye(r)
        try:
            V_new[b, :] = np.linalg.solve(ATA_reg, ATb)
        except np.linalg.LinAlgError:
            V_new[b, :] = np.linalg.lstsq(ATA_reg, ATb, rcond=None)[0]
    
    return V_new

def _als_step_W(U, V, W, T, n, m, p):
    """ALS update for W."""
    n, m, p = DIMENSIONS
    r = W.shape[1]
    W_new = np.zeros_like(W)
    
    for c in range(p*n):
        ATA = np.zeros((r, r), dtype=np.float64)
        ATb = np.zeros(r, dtype=np.float64)
        
        for a in range(n*m):
            for b in range(m*p):
                uv = U[a, :] * V[b, :]  # shape (r,)
                ATA += np.outer(uv, uv)
                if T[a, b, c] != 0:
                    ATb += T[a, b, c] * uv
        
        lam = 1e-8
        ATA_reg = ATA + lam * np.eye(r)
        try:
            W_new[c, :] = np.linalg.solve(ATA_reg, ATb)
        except np.linalg.LinAlgError:
            W_new[c, :] = np.linalg.lstsq(ATA_reg, ATb, rcond=None)[0]
    
    return W_new

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify exact tensor decomposition."""
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    T_rec = np.einsum('ak,bk,ck->abc', U, V, W)
    return np.max(np.abs(T_rec - T)) < tol

def _try_rank_als(rank, max_iter=50, deadline=None):
    """Try to find a rank-r decomposition using ALS."""
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    best_err = float('inf')
    best_solution = None
    
    for trial in range(5):
        if deadline is not None and time.time() > deadline - 30:
            break
        
        np.random.seed(trial * 7919 + rank)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline - 15:
                break
            
            U = _als_step_U(U, V, W, T, n, m, p)
            V = _als_step_V(U, V, W, T, n, m, p)
            W = _als_step_W(U, V, W, T, n, m, p)
            
            err = _tensor_error(U, V, W, T)
            if err < best_err:
                best_err = err
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                break
    
    if best_solution is not None and best_err < 1e-8:
        U, V, W = best_solution
        if _verify_decomposition(U, V, W, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

# EVOLVE-BLOCK-END


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try to find a better decomposition than schoolbook (rank 54)
    # Try various ranks from low to high
    for trial_rank in [10, 15, 20, 27, 30, 36, 40, 45, 50]:
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_als(trial_rank, max_iter=30, deadline=deadline)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
