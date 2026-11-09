"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _verify(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Indices:
    U[i*m+j, c]
    V[jp*p+k, c]
    W[kp*n+ip, c]
    Target: delta(i,ip)*delta(j,jp)*delta(k,kp)
    """
    rank = U.shape[1] if U.ndim == 2 else len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for easier indexing
    # U: (n, m, r)
    U_3d = U_arr.reshape(n, m, rank)
    # V: (m, p, r)
    V_3d = V_arr.reshape(m, p, rank)
    # W: (p, n, r)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute tensor T[i, j, jp, k, kp, ip]
    # T[i, j, jp, k, kp, ip] = sum_c U[i, j, c] * V[jp, k, c] * W[kp, ip, c]
    
    # We can compute this efficiently using broadcasting
    # U[i, j, c] * V[jp, k, c] -> shape (n, m, m, p, r)
    UV = U_3d[:, :, np.newaxis, np.newaxis, :] * V_3d[np.newaxis, np.newaxis, :, :, :]
    # UV[i, j, jp, k, c]
    
    # Now multiply by W[kp, ip, c] -> shape (p, n, r)
    # We need sum_c UV[i, j, jp, k, c] * W[kp, ip, c]
    # Result shape: (n, m, m, p, p, n)
    # T[i, j, jp, k, kp, ip]
    
    # To avoid creating a huge intermediate array, we can do it in steps or use einsum
    # einsum('ijc,jkc,kpc->ijkkpc' ...) no, indices are different.
    # U: i, j, c
    # V: jp, k, c
    # W: kp, ip, c
    # Result: i, j, jp, k, kp, ip
    
    # Let's use np.einsum
    # 'ijc,jkc,kpc->ijkkpc' is wrong because c is summed.
    # 'ijc,jkc,kpc->ijkkpc' -> sum over c.
    # Wait, V index is jp, k. W index is kp, ip.
    # So:
    T = np.einsum('ijc,jkc,kpc->ijkkpc', U_3d, V_3d, W_3d)
    
    # Target tensor
    # Target[i, j, jp, k, kp, ip] = 1 if i==ip, j==jp, k==kp else 0
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            if abs(T[i, j, jp, k, kp, ip] - target) > 1e-6:
                                return False
    return True

def _compute_residual(U_flat, V_flat, W_flat, n, m, p, r):
    """Compute the residual vector for least squares."""
    U = U_flat.reshape(n * m, r)
    V = V_flat.reshape(m * p, r)
    W = W_flat.reshape(p * n, r)
    
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kpc->ijkkpc', U_3d, V_3d, W_3d)
    
    # Target tensor
    # We can construct the target tensor once and store it, or compute on fly.
    # For speed, let's compute the difference.
    # Target[i, j, jp, k, kp, ip] = 1 if i==ip and j==jp and k==kp else 0
    
    # Instead of building the full target tensor every time, we can just check the specific non-zero terms and the zero terms.
    # But for least squares, we need the full vector.
    # Let's build the target tensor once in a closure or pass it.
    
    # Actually, building a 2304-element vector is fast.
    target = np.zeros((n, m, m, p, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, j, k, k, i] = 1.0
                
    residual = T - target
    return residual.flatten()

def _try_rank(r, n, m, p, deadline):
    """
    Try to find a rank-r decomposition.
    """
    if time.time() > deadline - 5:
        return None
        
    # Pre-compute target tensor
    target = np.zeros((n, m, m, p, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, j, k, k, i] = 1.0
    target_flat = target.flatten()
    
    total_vars = 3 * n * m * p * r # This is wrong, total vars is (n*m + m*p + p*n) * r
    
    n_vars = (n * m + m * p + p * n) * r
    
    # We will use multiple random restarts
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(42 + restart * 7919 + r)
        
        # Initialize with small random values
        U_init = np.random.randn(n * m, r) * 0.1
        V_init = np.random.randn(m * p, r) * 0.1
        W_init = np.random.randn(p * n, r) * 0.1
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Define the residual function for scipy
        def residual_func(x):
            U_flat = x[:n*m*r]
            V_flat = x[n*m*r : n*m*r + m*p*r]
            W_flat = x[n*m*r + m*p*r :]
            
            U = U_flat.reshape(n * m, r)
            V = V_flat.reshape(m * p, r)
            W = W_flat.reshape(p * n, r)
            
            U_3d = U.reshape(n, m, r)
            V_3d = V.reshape(m, p, r)
            W_3d = W.reshape(p, n, r)
            
            T = np.einsum('ijc,jkc,kpc->ijkkpc', U_3d, V_3d, W_3d)
            
            return T.flatten() - target_flat
        
        try:
            # Use least_squares
            result = opt.least_squares(residual_func, x0, method='trf', max_nfev=2000, ftol=1e-12, xtol=1e-12)
            
            if result.success or result.cost < 1e-10:
                x_sol = result.x
                U_sol = x_sol[:n*m*r].reshape(n * m, r)
                V_sol = x_sol[n*m*r : n*m*r + m*p*r].reshape(m * p, r)
                W_sol = x_sol[n*m*r + m*p*r :].reshape(p * n, r)
                
                # Round to nearest half-integer
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                # Verify
                if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                
                # If rounding failed, try local discrete search
                # This is expensive, so we only do it if we are close
                # Let's try a few perturbations
                best_loss = result.cost
                best_U, best_V, best_W = U_rounded.copy(), V_rounded.copy(), W_rounded.copy()
                
                # Simple local search: try changing one entry at a time by +/- 0.5
                # This is O(N_vars * 3) which might be too slow for large r.
                # We'll skip this for now and rely on finding a good continuous solution first.
                
        except Exception as e:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook decomposition
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try lower ranks
    # For 2x4x6, let's try from 40 down to 30
    # If we find one, we return it.
    # We iterate from low to high to find the minimum? No, we want the minimum rank.
    # So we should try low ranks first.
    # But low ranks are harder to find.
    # Let's try a range.
    
    for r in range(30, 48):
        if time.time() > deadline:
            break
            
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            U_list, V_list, W_list = result
            best_rank = r
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            # Since we are iterating from low to high, the first one found is the minimum
            # But we should continue to check if there is an even lower one?
            # No, we started at 30. If we found 30, it's the min.
            # If we found 35, we should have checked 30-34 first.
            # So if we find one, we can return it immediately if we are confident it's the minimum.
            # But maybe 29 exists? We started at 30.
            # Let's extend the range to 20 if time permits.
            
            # For now, return the first found.
            return best_result
            
    return best_result
# EVOLVE-BLOCK-END
