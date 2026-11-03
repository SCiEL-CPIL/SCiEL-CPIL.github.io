"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _build_rank55():
    """Build the rank-55 Strassen-based decomposition."""
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 55
    
    U = np.zeros((n * m, rank))  # (6, 55)
    V = np.zeros((m * p, rank))  # (30, 55)
    W = np.zeros((p * n, rank))  # (20, 55)
    
    col = 0
    
    # Part 1: A_right (A[:,2], shape 2x1) * B_bottom (B[2,:], shape 1x10)
    for i in range(n):
        a_idx = i * m + 2  # 2 or 5
        for k in range(p):
            b_idx = 2 * p + k  # 20..29
            c_idx = k * n + i  # 2k+i
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    # Part 2: A_left (A[:,:2], shape 2x2) * B_top (B[:2,:], shape 2x10)
    X00, X01, X10, X11 = 0, 1, 3, 4
    
    for j in range(5):
        Y00 = 2 * j
        Y01 = 2 * j + 1
        Y10 = 10 + 2 * j
        Y11 = 10 + 2 * j + 1
        Z00 = 4 * j
        Z10 = 4 * j + 1
        Z01 = 4 * j + 2
        Z11 = 4 * j + 3
        
        # M1 = (X00 + X11) * (Y00 + Y11)
        U[X00, col] += 1
        U[X11, col] += 1
        V[Y00, col] += 1
        V[Y11, col] += 1
        W[Z00, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M2 = (X10 + X11) * Y00
        U[X10, col] += 1
        U[X11, col] += 1
        V[Y00, col] += 1
        W[Z10, col] += 1
        W[Z11, col] -= 1
        col += 1
        
        # M3 = X00 * (Y01 - Y11)
        U[X00, col] += 1
        V[Y01, col] += 1
        V[Y11, col] -= 1
        W[Z01, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M4 = X11 * (Y10 - Y00)
        U[X11, col] += 1
        V[Y10, col] += 1
        V[Y00, col] -= 1
        W[Z00, col] += 1
        W[Z10, col] += 1
        col += 1
        
        # M5 = (X00 + X01) * Y11
        U[X00, col] += 1
        U[X01, col] += 1
        V[Y11, col] += 1
        W[Z00, col] -= 1
        W[Z01, col] += 1
        col += 1
        
        # M6 = (X10 - X00) * (Y00 + Y01)
        U[X10, col] += 1
        U[X00, col] -= 1
        V[Y00, col] += 1
        V[Y01, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M7 = (X01 - X11) * (Y10 + Y11)
        U[X01, col] += 1
        U[X11, col] -= 1
        V[Y10, col] += 1
        V[Y11, col] += 1
        W[Z00, col] += 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W


def _build_target_tensor():
    """Build the true matrix multiplication tensor T[i,j,k] for 2x3 * 3x10."""
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _als_reduce(U, V, W, T, target_rank, max_iters=200, tol=1e-10, deadline=None):
    """
    ALS optimization to reduce rank.
    U: (nm, R), V: (mp, R), W: (pn, R)
    T: (nm, mp, pn) target tensor
    Returns (U_new, V_new, W_new, error) or None if failed.
    """
    nm, mp, pn = T.shape
    R = target_rank
    
    # Initialize: take first R columns from the rank-55 decomposition
    # and add small random perturbation to escape local minima
    U_curr = U[:, :R].copy()
    V_curr = V[:, :R].copy()
    W_curr = W[:, :R].copy()
    
    # Add small random perturbation
    rng = np.random.RandomState(42)
    U_curr += rng.randn(*U_curr.shape) * 0.01
    V_curr += rng.randn(*V_curr.shape) * 0.01
    W_curr += rng.randn(*W_curr.shape) * 0.01
    
    prev_error = np.inf
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Update U: for each i, solve V^T * W^T (element-wise) * T[i] = U[i]
        # T[i] is a (mp, pn) matrix
        # We want: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
        # For fixed i: sum_r U[i,r] * (V[:,r] * W[:,r])^T = T[i]
        # This is: V * diag(U[i]) * W^T = T[i]
        # In matrix form: let M_i = V * W^T (mp x pn), then we need
        # sum_r U[i,r] * (V[:,r] * W[:,r]) = vec(T[i])
        # Better: use the matricization approach
        # Fix V and W, update U:
        # For each i: U[i,:] = argmin sum_{j,k} (T[i,j,k] - sum_r U[i,r]*V[j,r]*W[k,r])^2
        # = argmin ||T[i] - V * diag(U[i]) * W^T||_F^2
        # This is a linear system in U[i]
        # vec(T[i]) = (W ⊗ V) * vec(U[i])  -- but this is too large
        # Alternative: use the fact that T[i] = V * diag(U[i]) * W^T
        # So T[i] * W / W^T W = V * diag(U[i])
        # Then U[i,r] = (V^T * T[i] * W / W^T W * V)_r / ||V[:,r]||^2 ... not quite
        
        # Let's use the standard ALS approach:
        # Fix V, W. Update U:
        # T[i] = V * diag(U[i]) * W^T
        # => T[i] * W * (W^T W)^{-1} = V * diag(U[i])
        # => For each r: U[i,r] = (V[:,r]^T * T[i] * W * (W^T W)^{-1})_r / ||V[:,r]||^2
        # Wait, that's not right either. Let me think more carefully.
        
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Fix V, W. For fixed i:
        # T[i,j,k] = sum_r U[i,r] * (V[j,r] * W[k,r])
        # This is: T[i] = sum_r U[i,r] * outer(V[:,r], W[:,r])
        # In matrix form: T[i] = V * diag(U[i]) * W^T
        # So: T[i] * W * (W^T W)^{-1} = V * diag(U[i])
        # Let C = W^T W (R x R), C_inv = inv(C)
        # Then: V * diag(U[i]) = T[i] * W * C_inv
        # Taking element (j,r) of both sides:
        # V[j,r] * U[i,r] = (T[i] * W * C_inv)[j,r]
        # So: U[i,r] = (T[i] * W * C_inv)[j,r] / V[j,r] for any j where V[j,r] != 0
        # But this must hold for all j, so we need a least-squares solution:
        # min_U sum_j (V[j,r] * U[i,r] - (T[i]*W*C_inv)[j,r])^2
        # = U[i,r] = sum_j V[j,r] * (T[i]*W*C_inv)[j,r] / sum_j V[j,r]^2
        
        # Similarly for V and W.
        
        # Update U
        C_W = W_curr.T @ W_curr  # (R, R)
        C_W_inv = np.linalg.inv(C_W)
        # For each i:
        for i in range(nm):
            # T[i] is (mp, pn)
            # T[i] * W is (mp, R)
            # T[i] * W * C_W_inv is (mp, R)
            tmp = T[i] @ W_curr @ C_W_inv  # (mp, R)
            # U[i,r] = sum_j V_curr[j,r] * tmp[j,r] / sum_j V_curr[j,r]^2
            V_sq = np.sum(V_curr ** 2, axis=0)  # (R,)
            numerator = np.sum(V_curr * tmp, axis=0)  # (R,)
            U_curr[i, :] = numerator / V_sq
        
        # Update V
        C_U = U_curr.T @ U_curr  # (R, R)
        C_U_inv = np.linalg.inv(C_U)
        for j in range(mp):
            # T[:,j,:] is (nm, pn)
            # T[:,j,:] * W is (nm, R)  -- wait, need to be careful
            # T[:,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # Fix U, W. For fixed j:
            # T[:,j,k] = sum_r V[j,r] * (U[i,r] * W[k,r])
            # In matrix form: T[:,j,:] = U * diag(V[j]) * W^T
            # So: T[:,j,:] * W * (W^T W)^{-1} = U * diag(V[j])
            # Let C_W = W^T W, C_W_inv = inv(C_W)
            tmp = T[:, j, :] @ W_curr @ C_W_inv  # (nm, R)
            U_sq = np.sum(U_curr ** 2, axis=0)  # (R,)
            numerator = np.sum(U_curr * tmp, axis=0)  # (R,)
            V_curr[j, :] = numerator / U_sq
        
        # Update W
        C_V = V_curr.T @ V_curr  # (R, R)
        C_V_inv = np.linalg.inv(C_V)
        for k in range(pn):
            # T[:,:,k] is (nm, mp)
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # Fix U, V. For fixed k:
            # T[:,:,k] = U * diag(W[k]) * V^T
            # So: T[:,:,k] * V * (V^T V)^{-1} = U * diag(W[k])
            tmp = T[:, :, k] @ V_curr @ C_V_inv  # (nm, R)
            U_sq = np.sum(U_curr ** 2, axis=0)  # (R,)
            numerator = np.sum(U_curr * tmp, axis=0)  # (R,)
            W_curr[k, :] = numerator / U_sq
        
        # Compute error
        # Error = ||T - sum_r U[:,r] * outer(V[:,r], W[:,r])||_F^2
        # = ||T - V * diag(U) * W^T||_F^2 for each i... no, it's the full tensor
        # Let's compute it as: for each (i,j,k), error += (T[i,j,k] - sum_r U[i,r]*V[j,r]*W[k,r])^2
        # This is expensive. Use matrix form:
        # Reconstructed[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # = (V * diag(U[i]) * W^T)[j,k]
        # So for each i: recon[i] = V * diag(U[i]) * W^T
        # Error = sum_i ||T[i] - V * diag(U[i]) * W^T||_F^2
        
        error = 0.0
        for i in range(nm):
            recon_i = V_curr * U_curr[i, np.newaxis, :] @ W_curr.T  # (mp, pn)
            error += np.sum((T[i] - recon_i) ** 2)
        
        if abs(prev_error - error) < tol:
            break
        prev_error = error
    
    if deadline is not None and time.time() > deadline:
        return None
    
    return U_curr, V_curr, W_curr, error


def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 2x3 x 3x10 matrix multiplication.
    Starts from rank-55 Strassen-based decomposition and uses ALS to reduce rank.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the rank-55 decomposition
    U55, V55, W55 = _build_rank55()
    
    # Build the target tensor
    T = _build_target_tensor()
    
    # Verify the rank-55 decomposition is correct
    error55 = 0.0
    for i in range(n * m):
        recon_i = V55 * U55[i, np.newaxis, :] @ W55.T
        error55 += np.sum((T[i] - recon_i) ** 2)
    
    # Try to reduce rank from 55 down
    best_result = None
    best_rank = 55
    best_U, best_V, best_W = U55, V55, W55
    
    for target_rank in range(54, 30, -1):
        if time.time() > deadline:
            break
        
        # Try ALS with this target rank
        result = _als_reduce(U55, V55, W55, T, target_rank, 
                           max_iters=100, tol=1e-10, deadline=deadline)
        if result is not None:
            U_r, V_r, W_r, error = result
            # Check if the decomposition is exact (or nearly exact)
            if error < 1e-6:
                # Verify by reconstructing
                recon_error = 0.0
                for i in range(n * m):
                    recon_i = V_r * U_r[i, np.newaxis, :] @ W_r.T
                    recon_error += np.sum((T[i] - recon_i) ** 2)
                if recon_error < 1e-6:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    break  # Found a lower rank, stop
    
    # If no lower rank found, return rank-55
    U_out = best_U
    V_out = best_V
    W_out = best_W
    
    # Round to nearest half-integer if close (for cleaner output)
    def round_half(x):
        return np.round(x * 2) / 2
    
    U_out = round_half(U_out)
    V_out = round_half(V_out)
    W_out = round_half(W_out)
    
    # Final verification
    final_error = 0.0
    for i in range(n * m):
        recon_i = V_out * U_out[i, np.newaxis, :] @ W_out.T
        final_error += np.sum((T[i] - recon_i) ** 2)
    
    if final_error > 1e-4:
        # Rounding broke it, use unrounded
        U_out = best_U
        V_out = best_V
        W_out = best_W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
