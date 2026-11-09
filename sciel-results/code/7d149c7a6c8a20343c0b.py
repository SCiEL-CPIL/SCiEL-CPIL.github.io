import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R
    
    U_fb, V_fb, W_fb, R_fb = schoolbook()
    
    def compute_error(U, V, W, R):
        """Compute tensor reconstruction error."""
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reshape for broadcasting
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # Target: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        
        # Reshape U to (n, m, R)
        U_3d = U.reshape(n, m, R)
        # Reshape V to (m, p, R)
        V_3d = V.reshape(m, p, R)
        # Reshape W to (p, n, R)
        W_3d = W.reshape(p, n, R)
        
        # We need to compute T[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Let's iterate over k to keep memory low? Or use einsum.
        # T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        # This creates a (3,3,7) array. Very fast.
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.sum((T - 1.0)**2)
        return err

    def snap_and_verify(U, V, W, R, tol=1e-6):
        """Snap to half-integers and verify exactness."""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        err = compute_error(U_snap, V_snap, W_snap, R)
        if err < tol:
            return U_snap, V_snap, W_snap
        return None

    def try_als(R, seed, max_iters=200, tol=1e-10):
        """Try vectorized ALS with given rank and seed."""
        np.random.seed(seed)
        # Initialize with small random values to avoid overflow
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        prev_err = float('inf')
        
        # Precompute indices or structures?
        # U_3d: (n, m, R)
        # V_3d: (m, p, R)
        # W_3d: (p, n, R)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            # For each i, j:
            # M[i,j] is (p, R) where M[i,j,k,r] = V[j,k,r] * W[k,i,r]
            # Solve M[i,j] u = 1_p
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            # W_3d is (p, n, R). We need W[k, i, r].
            # Let's transpose W_3d to (n, p, R) to align i and k?
            # W_3d[k, i, r] -> W_trans[i, k, r]
            W_trans = np.transpose(W_3d, (1, 0, 2)) # (n, p, R)
            
            # We want to compute M for all i, j.
            # M[i, j] has shape (p, R).
            # M[i, j, k, r] = V_3d[j, k, r] * W_trans[i, k, r]
            
            # V_3d is (m, p, R). Broadcast to (1, m, p, R) -> (1, 3, 7, R)
            # W_trans is (n, p, R). Broadcast to (n, 1, p, R) -> (3, 1, 7, R)
            # Product: (n, m, p, R)
            
            M = V_3d[None, :, :, :] * W_trans[:, None, :, :] # (3, 3, 7, R)
            
            # Now solve M[i,j] u = 1 for all i, j.
            # M is (3, 3, 7, R). Reshape to (9, 7, R).
            M_flat = M.reshape(-1, p, R) # (9, 7, R)
            
            # We need to solve A x = b for 9 systems.
            # A is (9, 7, R). b is (7,).
            # x will be (9, R).
            
            # Using np.linalg.lstsq in a loop is fine for 9 items.
            new_U_3d = np.zeros_like(U_3d)
            for idx in range(n * m):
                A = M_flat[idx] # (7, R)
                b = np.ones(p)
                # Solve A x = b
                # x = np.linalg.lstsq(A, b, rcond=None)[0]
                # But we need to be careful with singular matrices.
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    x = np.zeros(R)
                new_U_3d.reshape(-1, R)[idx] = x
            
            U = new_U_3d.reshape(n * m, R)
            
            # Update V
            # For each j, k:
            # M[j,k] is (n, R) where M[j,k,i,r] = U[i,j,r] * W[k,i,r]
            # Solve M[j,k] v = 1_n
            
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            # U_3d[i, j, r]. We need U[i, j, r].
            # W_3d[k, i, r]. We need W[k, i, r].
            
            # M[j, k] shape (n, R).
            # M[j, k, i, r] = U_3d[i, j, r] * W_3d[k, i, r]
            
            # U_3d is (n, m, R). Transpose to (m, n, R)?
            # We want to index by j first?
            # Let's swap axes of U_3d to (m, n, R).
            U_trans = np.transpose(U_3d, (1, 0, 2)) # (m, n, R)
            
            # W_3d is (p, n, R).
            
            # M = U_trans[j, i, r] * W_3d[k, i, r]
            # U_trans: (m, n, R) -> (m, 1, n, R) ? No.
            # We want M of shape (m, p, n, R).
            # U_trans[j, i, r] -> (m, 1, n, R)
            # W_3d[k, i, r] -> (1, p, n, R)
            
            M = U_trans[:, None, :, :] * W_3d[None, :, :, :] # (3, 7, 3, R)
            
            M_flat = M.reshape(-1, n, R) # (21, 3, R)
            
            new_V_3d = np.zeros_like(V_3d)
            for idx in range(m * p):
                A = M_flat[idx] # (3, R)
                b = np.ones(n)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    x = np.zeros(R)
                new_V_3d.reshape(-1, R)[idx] = x
            
            V = new_V_3d.reshape(m * p, R)
            
            # Update W
            # For each k, i:
            # M[k,i] is (m, R) where M[k,i,j,r] = U[i,j,r] * V[j,k,r]
            # Solve M[k,i] w = 1_m
            
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            # U_3d[i, j, r].
            # V_3d[j, k, r].
            
            # M[k, i] shape (m, R).
            # M[k, i, j, r] = U_3d[i, j, r] * V_3d[j, k, r]
            
            # U_3d: (n, m, R). We want to index by i.
            # V_3d: (m, p, R). We want to index by k.
            
            # M shape (p, n, m, R).
            # U_3d[i, j, r] -> (n, m, 1, R)? No.
            # We want M[k, i, j, r].
            # U_3d[i, j, r] -> (1, n, m, R)
            # V_3d[j, k, r] -> (p, 1, m, R)?
            # V_3d is (m, p, R). Transpose to (p, m, R).
            V_trans = np.transpose(V_3d, (1, 0, 2)) # (p, m, R)
            
            # U_3d: (n, m, R) -> (1, n, m, R)
            # V_trans: (p, m, R) -> (p, 1, m, R)
            
            M = U_3d[None, :, :, :] * V_trans[:, None, :, :] # (7, 3, 3, R)
            
            M_flat = M.reshape(-1, m, R) # (21, 3, R)
            
            new_W_3d = np.zeros_like(W_3d)
            for idx in range(p * n):
                A = M_flat[idx] # (3, R)
                b = np.ones(m)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    x = np.zeros(R)
                new_W_3d.reshape(-1, R)[idx] = x
            
            W = new_W_3d.reshape(p * n, R)
            
            err = compute_error(U, V, W, R)
            
            if err < tol:
                return U, V, W
            
            if err > prev_err:
                break
            prev_err = err
        
        return None

    # Try ranks from 54 upward
    best_rank = n * m * p
    best_factors = None
    
    for R in range(54, min(63, best_rank)):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            result = try_als(R, seed=42 + attempt * 17, max_iters=150)
            if result is None:
                continue
            
            U, V, W = result
            snapped = snap_and_verify(U, V, W, R)
            if snapped is not None:
                U_s, V_s, W_s = snapped
                best_rank = R
                best_factors = (U_s, V_s, W_s)
                found = True
                break
        
        if found:
            break

    if best_factors:
        U, V, W = best_factors
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
