"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import numpy as np
import time
import random
import scipy.optimize as opt

DIMENSIONS = [2, 3, 4]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W exactly decompose the matrix multiplication tensor.
    Full tensor condition: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] 
    == delta_{i,i'} delta_{j,j'} delta_{k,k'}
    """
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                            val = 0.0
                            for r in range(rank):
                                val += U[i * m + j, r] * V[j2 * p + k, r] * W[k2 * n + i2, r]
                            if abs(val - expected) > tol:
                                return False
    return True

def _snap_to_half_int(arr):
    return np.round(np.asarray(arr, dtype=float) * 2) / 2.0

def _als_step(U, V, W, n, m, p, rank, max_iter=100, tol=1e-10):
    """
    Alternating Least Squares for tensor decomposition.
    For each mode, solve a least squares problem holding the other two fixed.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for iteration in range(max_iter):
        old_U, old_V, old_W = U.copy(), V.copy(), W.copy()
        
        # Update U: For each (i, j), solve for U[i*m+j, :] 
        # The constraint is: for each (i, j, i2, j2, k, k2),
        # sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i2, r] = delta_{i,i2}delta_{j,j2}delta_{k,k2}
        # This is a coupled system. Let's use the standard ALS approach:
        # Fix V and W, solve for U.
        
        # For each column r of U, we have constraints from all (i2, j2, k2) combinations.
        # Actually, the standard approach for matrix multiplication tensor:
        # The tensor T has entries T[i,j,j',k',i',k] where the multiplication is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k,j',i',k'] but let's be careful with indices.
        
        # Let me reformulate. The matrix multiplication tensor has:
        # T[i, j, k] (input) and we want C[i,k] = sum_j A[i,j] B[j,k]
        # So the decomposition is: C[i,k] = sum_r U[i,j,r] V[j,k,r] W[r] ... no.
        
        # Standard form: C[i,k] = sum_{j,r} A[i,j] B[j,k] W_{i,k,r} * U_{i,j,r} * V_{j,k,r}
        # Wait, the interface says:
        # U has shape (n*m, rank) - indexed by (i, j)
        # V has shape (m*p, rank) - indexed by (j, k)  
        # W has shape (p*n, rank) - indexed by (k, i)
        
        # And the condition is:
        # For fixed i, k: M[j, j'] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] should be Identity.
        
        # This means: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
        # for each i, k.
        
        # So for fixed i, k, we have:
        # sum_r (U[i*m+j, r] * W[k*n+i, r]) * V[j'*p+k, r] = delta_{j,j'}
        
        # Let's define for fixed i, k:
        # A_{i,k}[j, r] = U[i*m+j, r] * W[k*n+i, r]
        # Then: A_{i,k} @ V[:, k, :]^T = I_m (where V[:, k, :] is V[j*p+k, :] for j=0..m-1)
        
        # Actually V[j'*p+k, r] for fixed k is a vector of length m (over j').
        # Let V_k[j', r] = V[j'*p + k, r]
        # Then the condition is: for each i, k:
        # sum_r A_{i,k}[j, r] * V_k[j', r] = delta_{j,j'}
        # i.e., A_{i,k} @ V_k^T = I_m
        
        # Similarly, we can derive conditions for other modes.
        
        # For ALS, let's fix V and W, solve for U.
        # The constraint involves U[i*m+j, r] multiplied by V[j'*p+k, r] * W[k*n+i, r].
        
        # Let's use a different approach: build the full constraint matrix.
        # Number of constraints: n * m * m * p * p * n = n^2 * m^2 * p^2
        # For n=2, m=3, p=4: 4 * 9 * 16 = 576 constraints
        # Number of unknowns in U: n * m * rank = 6 * rank
        # For rank=24: 144 unknowns, 576 constraints - overdetermined
        
        # Actually, for ALS to work properly, I should fix two modes and solve for the third.
        # Let me fix V and W, and solve for U.
        
        # The constraint is: for all (i, j, j2, k, k2, i2):
        # sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i2, r] = delta_{i,i2}delta_{j,j2}delta_{k,k2}
        
        # For fixed U row (i, j), the unknowns are U[i*m+j, 0:rank]
        # But the constraint couples different U rows through the same r index.
        
        # This is complex. Let me use a simpler approach: 
        # Use the fact that for fixed i, k, the matrix M = sum_r outer(U[i*m:(i+1)*m, r], V[:, k, r]) * W[k*n+i, r] should be I.
        
        # Actually, let me just use scipy's least_squares but with better initialization and more restarts.
        # Or implement a simple gradient descent.
        
        # Let me try a different strategy: use the structure more directly.
        # For the matrix multiplication tensor, there are known decompositions.
        # The rank of 2x3*3x4 is at most 24 (schoolbook).
        # Is it less? The Strassen-like methods might help, but for these small dimensions,
        # the rank might be exactly 24 or slightly less.
        
        # Let me just improve the current approach with better initialization.
        
        break  # Skip ALS for now, use improved least_squares
    
    return U, V, W

def _try_rank(n, m, p, target_rank, time_limit=5):
    start_time = time.time()
    best_residual = float('inf')
    best_params = None
    
    # Residual function using vectorized operations for speed
    def residuals(params):
        U = params[:n*m*target_rank].reshape(n*m, target_rank)
        V = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        # Vectorized computation of all tensor entries
        # T[i,j,j2,k,k2,i2] = sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i2, r]
        
        # Build index arrays
        i_idx = np.repeat(np.arange(n), m * m * p * p * n)
        j_idx = np.tile(np.repeat(np.arange(m), m * p * p * n), n)
        j2_idx = np.tile(np.repeat(np.arange(m), p * p * n), n * m)
        k_idx = np.tile(np.repeat(np.arange(p), p * n), n * m * m)
        k2_idx = np.tile(np.repeat(np.arange(p), n), n * m * m * p)
        i2_idx = np.tile(np.arange(n), n * m * m * p * p)
        
        # Extract values
        U_vals = U[i_idx * m + j_idx]  # shape (num_constraints, target_rank)
        V_vals = V[j2_idx * p + k_idx]  # shape (num_constraints, target_rank)
        W_vals = W[k2_idx * n + i2_idx]  # shape (num_constraints, target_rank)
        
        # Compute tensor entries: sum over r
        T_vals = np.sum(U_vals * V_vals * W_vals, axis=1)
        
        # Expected values
        expected = np.zeros(len(i_idx))
        for idx in range(len(i_idx)):
            if i_idx[idx] == i2_idx[idx] and j_idx[idx] == j2_idx[idx] and k_idx[idx] == k2_idx[idx]:
                expected[idx] = 1.0
        
        return T_vals - expected

    for restart in range(30):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization with different scales
        scale = np.random.choice([0.3, 0.5, 1.0, 2.0])
        U_flat = np.random.randn(n * m * target_rank) * scale
        V_flat = np.random.randn(m * p * target_rank) * scale
        W_flat = np.random.randn(p * n * target_rank) * scale
        
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        # Try LM
        try:
            res = opt.least_squares(residuals, x0, method='lm', max_nfev=3000)
            if res.cost < best_residual:
                best_residual = res.cost
                best_params = res.x.copy()
                if best_residual < 1e-10:
                    break
        except Exception:
            pass
            
        # Try TRF
        try:
            res2 = opt.least_squares(residuals, x0, method='trf', max_nfev=3000)
            if res2.cost < best_residual:
                best_residual = res2.cost
                best_params = res2.x.copy()
                if best_residual < 1e-10:
                    break
        except Exception:
            pass

    if best_params is not None and best_residual < 1e-8:
        U = best_params[:n*m*target_rank].reshape(n*m, target_rank)
        V = best_params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W = best_params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        U_snapped = _snap_to_half_int(U)
        V_snapped = _snap_to_half_int(V)
        W_snapped = _snap_to_half_int(W)
        
        if _verify(U_snapped, V_snapped, W_snapped, n, m, p):
            return U_snapped, V_snapped, W_snapped
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = 115.0
    start_time = time.time()
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try ranks from 23 down to 18
    for r in range(best_rank - 1, max(15, best_rank - 6), -1):
        if time.time() > deadline - 5:
            break
        
        U, V, W = _try_rank(n, m, p, r, time_limit=12)
        if U is not None:
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
    
    if best_U is None:
        best_U, best_V, best_W = _schoolbook(n, m, p)
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
