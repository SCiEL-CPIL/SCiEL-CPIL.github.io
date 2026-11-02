import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 120
    
    # Deadline guard
    start_time = time.time()
    deadline = 145  # 5s safety margin from 150s limit
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = make_schoolbook()
    best_rank = rank_school
    
    # Target tensor T[i,j,k] = 1
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Dimensions for U, V, W
    dim_U = n * m  # 20
    dim_V = m * p  # 30
    dim_W = p * n  # 24
    
    def try_rank(R, max_restarts=5):
        """Try to find a rank-R decomposition."""
        if time.time() > deadline:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with schoolbook terms (first R terms)
            x0 = np.zeros(R * (dim_U + dim_V + dim_W))
            
            count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if count >= R:
                            break
                        r = count
                        offset = r * (dim_U + dim_V + dim_W)
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        x0[offset + u_idx] = 1.0
                        x0[offset + dim_U + v_idx] = 1.0
                        x0[offset + dim_U + dim_V + w_idx] = 1.0
                        count += 1
                        if count >= R:
                            break
                    if count >= R:
                        break
                if count >= R:
                    break
            
            def residual(x):
                U_flat = x[0:R * dim_U]
                V_flat = x[R * dim_U:R * dim_U + R * dim_V]
                W_flat = x[R * dim_U + R * dim_V:]
                
                U = U_flat.reshape(R, dim_U)
                V = V_flat.reshape(R, dim_V)
                W = W_flat.reshape(R, dim_W)
                
                # Reshape to 3D: U[r, i*m+j] -> U_3d[i, j, r]
                U_3d = U.T.reshape(n, m, R)
                V_3d = V.T.reshape(m, p, R)
                W_3d = W.T.reshape(p, n, R)
                
                # Sum over r: T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
                tensor_sum = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                
                res = tensor_sum - T
                return res.flatten()
            
            try:
                from scipy.optimize import least_squares
                res = least_squares(residual, x0, method='lm', max_nfev=5000)
                
                if res.cost < 1e-10:
                    # Extract solution
                    U_flat = res.x[0:R * dim_U]
                    V_flat = res.x[R * dim_U:R * dim_U + R * dim_V]
                    W_flat = res.x[R * dim_U + R * dim_V:]
                    
                    U = U_flat.reshape(dim_U, R)
                    V = V_flat.reshape(dim_V, R)
                    W = W_flat.reshape(dim_W, R)
                    
                    # Try to round to half-integers
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    
                    # Verify the rounded solution
                    U_3d_h = U_half.T.reshape(n, m, R)
                    V_3d_h = V_half.T.reshape(m, p, R)
                    W_3d_h = W_half.T.reshape(p, n, R)
                    tensor_sum_h = np.einsum('ijr,jkr,kir->ijk', U_3d_h, V_3d_h, W_3d_h)
                    error_h = np.linalg.norm(tensor_sum_h - T)
                    
                    if error_h < 1e-9:
                        return U_half.tolist(), V_half.tolist(), W_half.tolist(), R
            except Exception:
                pass
        
        return None
    
    # Try ranks from 119 down to 100
    for R in range(rank_school - 1, 99, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(R, max_restarts=3)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            # Continue trying to find even lower rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
