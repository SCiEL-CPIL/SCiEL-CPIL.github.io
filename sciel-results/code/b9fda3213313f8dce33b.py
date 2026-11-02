import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize Schoolbook
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    current_rank = rank_schoolbook
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = current_rank
    
    start_time = time.time()
    time_limit = 135  # Safety margin: 15s before 150s kill
    
    def compute_tensor(U, V, W, r):
        """Compute the full tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
        T = np.zeros((n, m, p))
        for c in range(r):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            prod_uv = u_c[:, :, None] * v_c[None, :, :]
            w_tensor = w_c.T[:, None, :]
            T += prod_uv * w_tensor
        return T

    def is_valid(U, V, W, r, atol=1e-6):
        T = compute_tensor(U, V, W, r)
        return np.allclose(T, 1.0, atol=atol)

    def snap_to_int(matrix):
        """Snap to nearest integer"""
        return np.round(matrix)

    def snap_to_half_int(matrix):
        """Snap to nearest half-integer"""
        return np.round(matrix * 2) / 2

    def try_snap_and_validate(U_new, V_new, W_new, r_new, snap_func):
        """Try snapping and validating"""
        U_snapped = snap_func(U_new)
        V_snapped = snap_func(V_new)
        W_snapped = snap_func(W_new)
        if is_valid(U_snapped, V_snapped, W_snapped, r_new):
            return U_snapped, V_snapped, W_snapped
        return None

    def try_reduce_rank(U, V, W, r):
        """
        Attempts to reduce rank by 1 by removing one term and solving for the rest.
        Returns (new_U, new_V, new_W, new_rank) or None.
        """
        # Try removing each term
        for c_remove in range(r):
            if time.time() - start_time > time_limit:
                return None
            
            # Create indices of terms to keep
            keep_idx = [i for i in range(r) if i != c_remove]
            
            # Extract current matrices for kept terms
            U_keep = U[:, keep_idx]
            V_keep = V[:, keep_idx]
            W_keep = W[:, keep_idx]
            
            # Compute current sum T_keep
            T_keep = compute_tensor(U_keep, V_keep, W_keep, r - 1)
            
            b = 1.0 - T_keep
            b_flat = b.flatten() # Size n*m*p = 105
            
            # Build Jacobian J
            N_u = n * m
            N_v = m * p
            N_w = p * n
            N_vars = (N_u + N_v + N_w) * (r - 1)
            
            J = np.zeros((n * m * p, N_vars))
            
            # For each equation (i, j, k):
            # d T[i,j,k] / d U[i,j,c] = V[j,k,c] * W[k,i,c]
            # d T[i,j,k] / d V[j,k,c] = U[i,j,c] * W[k,i,c]
            # d T[i,j,k] / d W[k,i,c] = U[i,j,c] * V[j,k,c]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        
                        for idx, c in enumerate(keep_idx):
                            # U derivative
                            val_U = V_keep[j * p + k, idx] * W_keep[k * n + i, idx]
                            var_idx_U = i * m + j
                            J[eq_idx, var_idx_U * (r - 1) + idx] = val_U
                            
                            # V derivative
                            val_V = U_keep[i * m + j, idx] * W_keep[k * n + i, idx]
                            var_idx_V = j * p + k
                            J[eq_idx, (N_u * (r - 1)) + var_idx_V * (r - 1) + idx] = val_V
                            
                            # W derivative
                            val_W = U_keep[i * m + j, idx] * V_keep[j * p + k, idx]
                            var_idx_W = k * n + i
                            J[eq_idx, (N_u + N_v) * (r - 1) + var_idx_W * (r - 1) + idx] = val_W
            
            # Solve J delta = b_flat
            try:
                delta_p, residuals, rank_J, sv = np.linalg.lstsq(J, b_flat, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            # Check if residual is small
            if np.linalg.norm(J.dot(delta_p) - b_flat) > 1e-5:
                continue
            
            # Compute SVD for null space
            U_svd, s, Vh = np.linalg.svd(J, full_matrices=False)
            
            # Try different strategies to find a valid snapped solution
            found = False
            
            # Strategy 1: Try the min-norm solution directly with integer snap
            U_new = U_keep + delta_p[:N_u * (r - 1)].reshape(N_u, r - 1)
            V_new = V_keep + delta_p[N_u * (r - 1): (N_u + N_v) * (r - 1)].reshape(N_v, r - 1)
            W_new = W_keep + delta_p[(N_u + N_v) * (r - 1):].reshape(N_w, r - 1)
            
            result = try_snap_and_validate(U_new, V_new, W_new, r - 1, snap_to_int)
            if result:
                return result[0], result[1], result[2], r - 1
            
            result = try_snap_and_validate(U_new, V_new, W_new, r - 1, snap_to_half_int)
            if result:
                return result[0], result[1], result[2], r - 1
            
            # Strategy 2: Try random null-space perturbations
            # Precompute projection matrix for null space
            # Null space projection: P_null = I - Vh.T @ Vh
            # But Vh is (105, N_vars), so Vh.T @ Vh is (N_vars, N_vars)
            # This is expensive, so we'll do it on the fly
            
            for attempt in range(50):
                if time.time() - start_time > time_limit:
                    return None
                
                # Generate random vector in null space
                z = np.random.randn(N_vars)
                z_row = Vh.T.dot(Vh.dot(z))
                z_null = z - z_row
                
                # Try different scales of the null-space perturbation
                for scale in [0.1, 0.5, 1.0, 2.0]:
                    delta = delta_p + scale * z_null
                    
                    U_new = U_keep + delta[:N_u * (r - 1)].reshape(N_u, r - 1)
                    V_new = V_keep + delta[N_u * (r - 1): (N_u + N_v) * (r - 1)].reshape(N_v, r - 1)
                    W_new = W_keep + delta[(N_u + N_v) * (r - 1):].reshape(N_w, r - 1)
                    
                    # Try integer snap first (more likely to work if entries are near integers)
                    result = try_snap_and_validate(U_new, V_new, W_new, r - 1, snap_to_int)
                    if result:
                        return result[0], result[1], result[2], r - 1
                    
                    result = try_snap_and_validate(U_new, V_new, W_new, r - 1, snap_to_half_int)
                    if result:
                        return result[0], result[1], result[2], r - 1
            
            if found:
                break
                
        return None

    # Main Loop
    while time.time() - start_time < time_limit:
        result = try_reduce_rank(best_U, best_V, best_W, best_rank)
        if result:
            best_U, best_V, best_W, best_rank = result
            # Try to reduce again
            continue
        else:
            # No reduction found
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
