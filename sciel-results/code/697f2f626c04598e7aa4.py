import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _tensor_error(U, V, W, n, m, p):
    """
    Compute the sum of squared errors for the tensor decomposition.
    The tensor T_{ijk} should be 1.
    T_{ijk} = sum_r U_{ir} V_{jr} W_{kr} (indices adjusted for matrix layout)
    U is (n*m, R), V is (m*p, R), W is (p*n, R)
    Index mapping:
    U[i, j] -> U[i*m + j]
    V[j, k] -> V[j*p + k]
    W[k, i] -> W[k*n + i]
    """
    R = U.shape[1]
    # We need to compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
    # and compare with 1.
    
    # To vectorize this efficiently, we can iterate over one dimension and use matrix products.
    # Let's fix i.
    # For fixed i, we need E_jk = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a bit tricky to fully vectorize without large intermediate arrays.
    # Given small sizes (3x4x4), a loop over i is cheap.
    
    total_err = 0.0
    for i in range(n):
        # U_i is a matrix of shape (m, R)
        U_i = U[i*m:(i+1)*m, :] 
        
        # We need to compute for all j, k:
        # val_{jk} = sum_r U_i[j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's reshape V and W to make access easier?
        # V is (m*p, R). V[j*p+k, r]
        # W is (p*n, R). W[k*n+i, r]
        
        # We can compute this by iterating over j? Or k?
        # Let's iterate over j.
        for j in range(m):
            # u_j is vector of shape (R,)
            u_j = U_i[j, :]
            
            # We need sum_r u_j[r] * V[j*p+k, r] * W[k*n+i, r] for all k in 0..p-1
            # Let v_jk = V[j*p+k, :] (shape R)
            # Let w_ki = W[k*n+i, :] (shape R)
            
            # Product element-wise: P_k = V[j*p+k, :] * W[k*n+i, :]
            # Then val_k = sum_r u_j[r] * P_k[r] = dot(u_j, P_k)
            
            # Vectorize over k:
            # V_j = V[j*p:(j+1)*p, :] shape (p, R)
            # W_i = W[i::n, :] ? No, W[k*n+i, :] for k=0..p-1.
            # Indices in W: i, n+i, 2n+i, ... (p-1)n+i
            # W_i = W[i:p*n:n, :] shape (p, R)
            
            V_j = V[j*p:(j+1)*p, :]
            W_i = W[i:p*n:n, :]
            
            # P = V_j * W_i (element-wise) shape (p, R)
            P = V_j * W_i
            
            # vals = P @ u_j shape (p,)
            vals = P @ u_j
            
            # Error for this j is sum_k (vals[k] - 1)^2
            err = np.sum((vals - 1.0)**2)
            total_err += err
            
    return total_err

def _gradient_descent(U, V, W, n, m, p, max_iter=2000, lr=0.01, deadline=None):
    """
    Perform gradient descent to minimize the squared error.
    """
    R = U.shape[1]
    best_err = _tensor_error(U, V, W, n, m, p)
    
    # Precompute indices for efficiency?
    # The error function is differentiable.
    # We can compute gradients analytically.
    
    # dE/dU_{im,jr} = sum_k 2 * (val_{ijk} - 1) * V_{jr,k} * W_{kr,i}
    # Note: V index is j*p+k, W index is k*n+i.
    
    # Let's implement a simple gradient step.
    # To speed up, we can compute the residual tensor R_{ijk} = val_{ijk} - 1.
    # Then gradients are contractions of R with the other factors.
    
    # Since n,m,p are small, we can afford to compute the full residual tensor (3x4x4 = 48 elements)
    # and then compute gradients via loops or vectorized contractions.
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 2:
            break
            
        # Compute residual tensor
        Res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                u_j = U[i*m+j, :]
                V_j = V[j*p:(j+1)*p, :]
                W_i = W[i:p*n:n, :]
                P = V_j * W_i
                vals = P @ u_j
                Res[i, j, :] = vals - 1.0
                
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # dU_{im, r} = sum_{j,k} 2 * Res[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
        # dV_{jm, r} = sum_{i,k} 2 * Res[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
        # dW_{kn, r} = sum_{i,j} 2 * Res[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
        
        # This is O(n*m*p*R) per gradient element, total O(n*m*p*R * (n*m + m*p + p*n))
        # 48 * 44 * 48 ~ 100k operations per step. 2000 steps ~ 200M ops. Might be slow in Python loops.
        # Let's try to vectorize the gradient computation slightly.
        
        # For dU:
        # dU[i*m+j, r] = 2 * sum_k Res[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
        # Let's loop over i, j.
        for i in range(n):
            for j in range(m):
                # Res_ij is vector of length p
                res_ij = Res[i, j, :]
                # V_j is (p, R)
                V_j = V[j*p:(j+1)*p, :]
                # W_i is (p, R)
                W_i = W[i:p*n:n, :]
                
                # We need sum_k res_ij[k] * V_j[k, r] * W_i[k, r]
                # This is sum_k (res_ij * (V_j * W_i))_k,r
                # Let M = V_j * W_i (p, R)
                # Then dU_row = sum_k res_ij[:, None] * M -> (p, R) @ (p, 1) ? No.
                # It's (res_ij * M).sum(axis=0)
                
                M = V_j * W_i
                dU[i*m+j, :] = 2.0 * (res_ij[:, None] * M).sum(axis=0)
                
        # For dV:
        # dV[j*p+k, r] = 2 * sum_i Res[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
        for j in range(m):
            for k in range(p):
                # We need sum_i Res[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                # U_col = U[0:m, j] ? No, U[i*m+j, :] for i=0..n-1.
                # Indices: j, m+j, 2m+j...
                U_j = U[j::m, :] # Shape (n, R)
                W_k = W[k::n, :] # Shape (p, R) -- Wait, W[k*n+i, :] for i=0..n-1.
                # W indices: k, n+k, 2n+k...
                # My previous W_i extraction was W[i:p*n:n, :].
                # Here we need W[k*n+i, :] for fixed k, varying i.
                # That is W[k::n, :]? No.
                # W is (p*n, R). Index is k*n + i.
                # For fixed k, i varies 0..n-1. Indices: k*n, k*n+1, ..., k*n+n-1.
                # So W_k = W[k*n:(k+1)*n, :] # Shape (n, R)
                
                U_j = U[j::m, :] # Shape (n, R)
                W_k = W[k*n:(k+1)*n, :] # Shape (n, R)
                
                # Res_col = Res[:, j, k] # Shape (n,)
                res_col = Res[:, j, k]
                
                M = U_j * W_k
                dV[j*p+k, :] = 2.0 * (res_col[:, None] * M).sum(axis=0)
                
        # For dW:
        # dW[k*n+i, r] = 2 * sum_j Res[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
        for k in range(p):
            for i in range(n):
                # We need sum_j Res[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
                # U_row = U[i*m:(i+1)*m, :] # Shape (m, R)
                # V_col = V[k::p, :]? No.
                # V index is j*p+k. For fixed k, j varies 0..m-1.
                # Indices: k, p+k, 2p+k...
                # So V_k = V[k::p, :] # Shape (m, R)
                
                U_row = U[i*m:(i+1)*m, :] # Shape (m, R)
                V_k = V[k::p, :] # Shape (m, R)
                
                # Res_row = Res[i, :, k] # Shape (m,)
                res_row = Res[i, :, k]
                
                M = U_row * V_k
                dW[k*n+i, :] = 2.0 * (res_row[:, None] * M).sum(axis=0)
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        lr *= 0.999
        
        # Check convergence
        curr_err = _tensor_error(U, V, W, n, m, p)
        if curr_err < best_err:
            best_err = curr_err
            
    return U, V, W, best_err

def _snap_to_half_integer(arr, tol=1e-4):
    """Snap array entries to nearest half-integer if close enough."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped, True
    return arr, False

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact for all (i,j,k)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_snap_and_check(U, V, W, n, m, p):
    """Try to snap to half-integers and check exactness."""
    for tol in [1e-3, 1e-2, 1e-1]:
        U_snapped, u_ok = _snap_to_half_integer(U, tol=tol)
        V_snapped, v_ok = _snap_to_half_integer(V, tol=tol)
        W_snapped, w_ok = _snap_to_half_integer(W, tol=tol)
        if u_ok and v_ok and w_ok:
            if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                return U_snapped, V_snapped, W_snapped
    return None

def _try_rank_gradient(n, m, p, rank, max_restarts=10, seed=None, deadline=None):
    """Try to find a rank-r exact decomposition using Gradient Descent."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    best_result = None
    best_err = float('inf')
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Random initialization
        # Scale is important. If rank is close to schoolbook, values should be small.
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Run gradient descent
        U, V, W, err = _gradient_descent(U, V, W, n, m, p, max_iter=1000, lr=0.05, deadline=deadline)
        
        if err < best_err:
            best_err = err
        
        # Try to snap
        if err < 1e-4: # Only try snapping if error is small enough
            result = _try_snap_and_check(U, V, W, n, m, p)
            if result is not None:
                return result
            
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    
    # Try ranks from 44 up to 47
    # 44 is the theoretical lower bound.
    # 48 is schoolbook.
    
    for rank in [44, 45, 46, 47]:
        if time.time() > deadline - 10:
            break
            
        # Increase restarts for higher ranks as they are more likely to exist?
        # Or just use fixed budget.
        # Let's allocate time based on remaining time.
        time_left = deadline - time.time()
        if time_left < 5:
            break
        
        # Try gradient descent
        result = _try_rank_gradient(n, m, p, rank, max_restarts=5, seed=rank, deadline=deadline)
        
        if result is not None:
            U, V, W = result
            print(f"Found rank {rank} decomposition")
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # If no better rank found, return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
