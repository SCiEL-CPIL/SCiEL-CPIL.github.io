import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    # Indices: i (0..n-1), j (0..m-1), j2 (0..m-1), k (0..p-1), i2 (0..n-1), k2 (0..p-1)
    # T[i, j, j2, k, i2, k2] = 1 if j==j2 and i==i2 and k==k2 else 0
    # Shape: (n, m, m, p, n, p)
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
                
    T_flat = T.flatten()
    
    def residual_fn(params, R):
        # params: flattened U, V, W
        # U: (n*m, R)
        # V: (m*p, R)
        # W: (p*n, R)
        
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape((n * m, R)).T # (R, n*m)
        V = V_flat.reshape((m * p, R)).T # (R, m*p)
        W = W_flat.reshape((p * n, R)).T # (R, p*n)
        
        # Reconstruct tensor
        # T_recon[i, j, j2, k, i2, k2] = sum_r U[r, i*m+j] * V[r, j2*p+k] * W[r, k2*n+i2]
        # We need to compute this efficiently.
        # Let's loop over r and add outer products?
        # Or use einsum?
        # np.einsum('r, r, r...') is not direct.
        
        # Construct a 3-tensor for each factor? No.
        # Let's just loop over R if R is small, or use broadcasting if possible.
        # U (R, n, m), V (R, m, p), W (R, p, n)
        
        U_reshaped = U.reshape((R, n, m))
        V_reshaped = V.reshape((R, m, p))
        W_reshaped = W.reshape((R, p, n))
        
        # We want T[i, j, j2, k, i2, k2] = sum_r U[r, i, j] V[r, j2, k] W[r, k2, i2]
        # Note: W index is k2*n + i2. So W[r, k2, i2].
        
        # This is a 6-tensor sum.
        # We can compute the residual by computing the full tensor and subtracting T_flat.
        
        # Efficient computation:
        # Sum over r of (U_r[i,j] * V_r[j2,k] * W_r[k2,i2])
        # This is equivalent to:
        # A = np.einsum('rij,rjk->riik', U_reshaped, V_reshaped) ? No.
        
        # Let's just do a loop over R for simplicity, as R < 84 and dims are small.
        # Total elements 1176. R*1176 operations. 80 * 1176 ~ 94k ops. Very fast.
        
        recon = np.zeros_like(T)
        for r in range(R):
            u = U_reshaped[r] # (n, m)
            v = V_reshaped[r] # (m, p)
            w = W_reshaped[r] # (p, n)
            
            # outer products
            # u[i, j] * v[j2, k] -> shape (n, m, m, p)
            uv = np.einsum('ij,jk->ijjk', u, v)
            # multiply by w[k2, i2] -> shape (n, m, m, p, p, n)
            # w is (p, n). Index k2, i2.
            # uv has shape (n, m, m, p) with indices (i, j, j2, k)
            # w has shape (p, n) with indices (k2, i2)
            # We want result[i, j, j2, k, k2, i2]
            # uv has shape (n, m, m, p) with indices (i, j, j2, k)
            # w has shape (p, n) with indices (k2, i2)
            # We want result[i, j, j2, k, k2, i2]
            # uv has shape (n, m, m, p) with indices (i, j, j2, k)
            # w has shape (p, n) with indices (k2, i2)
            # We want result[i, j, j2, k, k2, i2]
            recon += np.einsum('ijjk,ab->ijkab', uv, w)
            
        return (recon.flatten() - T_flat)
        
    best_solution = None
    best_rank = n * m * p
    
    # Known lower bound for matmul(3,4,7) is at least 3*4=12, but practically
    # start from a reasonable heuristic. For small dims, rank is often near n*p + m*p + n*m - ...
    # Let's start from max(n*p, m*p, n*m) + something, or just a heuristic like n*p + m - 1 etc.
    # For 3x4x7, schoolbook is 84. Strassen-like is lower.
    # Let's start from a lower bound estimate: max(n, m, p) * max(n, m, p) is too low.
    # A safe lower bound is max(n*p, m*p, n*m). For 3,4,7: max(21, 28, 12) = 28.
    # But we know matmul(2,2,2) is 7. matmul(3,3,3) is 23.
    # Let's just start from a reasonable range, say 50 to 83, to save time on impossible low ranks.
    # Actually, let's try to be smarter. The rank must be at least the rank of any slice.
    # But computing that is expensive. Let's just start from 50.
    
    # Generate schoolbook solution for warm-starting
    def get_schoolbook_solution(R_target=None):
        """Return schoolbook solution padded to R_target if needed, or full if R_target is None"""
        full_rank = n * m * p
        if R_target is None or R_target >= full_rank:
            R = full_rank
        else:
            R = R_target
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        for r in range(min(R, full_rank)):
            i = r // (m*p)
            j = (r // p) % m
            k = r % p
            U[i*m + j, r] = 1.0
            V[j*p + k, r] = 1.0
            W[k*n + i, r] = 1.0
            
        return U, V, W
    
    # Try ranks from 50 to 83
    for R in range(50, min(84, n*m*p)):
        if time.time() > deadline:
            break
            
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            total_size = size_U + size_V + size_W
            
            if restart == 0:
                # Warm start with schoolbook solution (padded with zeros)
                U_sb, V_sb, W_sb = get_schoolbook_solution(R)
                x0 = np.concatenate([U_sb.flatten(), V_sb.flatten(), W_sb.flatten()])
            else:
                # Random initialization
                x0 = np.random.randn(total_size) * 0.5
            
            try:
                res = least_squares(residual_fn, x0, args=(R,), method='lm', max_nfev=300)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                # Check if we can snap to half-integers
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U_snap = np.round(U_flat * 2) / 2
                V_snap = np.round(V_flat * 2) / 2
                W_snap = np.round(W_flat * 2) / 2
                
                # Verify snapped solution
                x_snapped = np.concatenate([U_snap, V_snap, W_snap])
                res_snapped = residual_fn(x_snapped, R)
                
                if np.max(np.abs(res_snapped)) < 1e-9:
                    # Valid!
                    best_rank = R
                    best_solution = {
                        "U": U_snap.reshape((n*m, R)).tolist(),
                        "V": V_snap.reshape((m*p, R)).tolist(),
                        "W": W_snap.reshape((p*n, R)).tolist()
                    }
                    break
        if best_solution:
            break

    if best_solution:
        return best_solution
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
