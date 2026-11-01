import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    target_rank = 11
    
    # Shapes
    dim_u = n * m  # 4
    dim_v = m * p  # 6
    dim_w = p * n  # 6
    
    def residual(flat_params):
        # Unpack
        U = flat_params[:dim_u * target_rank].reshape(dim_u, target_rank)
        V = flat_params[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
        W = flat_params[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
        
        # Compute tensor entries
        # T[i, j, j', k', i', k] = sum_r U[i, j, r] V[j', k', r] W[i', k, r]
        # We need this to be 1 if i==i', j==j', k==k' else 0.
        
        # Efficient computation:
        # Sum over r of U_{i,j,r} V_{j',k',r} W_{i',k,r}
        # This is a 3-mode tensor contraction.
        
        # We can compute this by iterating over r
        # Or using einsum
        # T = np.einsum('ir,kr,sr->is', U, W, V) ? No.
        # U is (i, j, r) -> index i*m+j
        # V is (j', k', r) -> index j'*p+k'
        # W is (i', k, r) -> index k*n+i'
        
        # Let's construct the full tensor T of shape (n, m, m, p, n, p)
        # T[i, j, jp, kp, ip, kp]
        
        # Using einsum:
        # U_flat is (n*m, rank). Reshape to (n, m, rank).
        # V_flat is (m*p, rank). Reshape to (m, p, rank).
        # W_flat is (p*n, rank). Reshape to (p, n, rank).
        
        U_3d = U.reshape(n, m, target_rank)
        V_3d = V.reshape(m, p, target_rank)
        W_3d = W.reshape(p, n, target_rank)
        
        # We want sum_r U[i,j,r] V[jp,kp,r] W[kp,ip,r]
        # Note: V index is j'*p+k'. So V_3d[j', k', r].
        # W index is k*n+i'. So W_3d[k, i', r].
        
        # Result T[i, j, j', k', i', k]
        # T = np.einsum('ijr,jkr,kr->ijjkik', U_3d, V_3d, W_3d)
        # Wait, indices:
        # U: i, j, r
        # V: j', k', r
        # W: k, i', r
        # Output: i, j, j', k', i', k
        
        T = np.einsum('ijr,jkr,kr->ijjkik', U_3d, V_3d, W_3d)
        
        # Target tensor
        # Target[i, j, j', k', i', k] = 1 if i==i', j==j', k==k' else 0
        
        # Create target
        target = np.zeros((n, m, m, p, n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i, j, j, k, i, k] = 1.0
        
        return (T - target).ravel()

    def snap_to_half_int(x):
        # Snap to nearest half-integer
        return np.round(x * 2) / 2.0

    def verify(U, V, W):
        n, m, p = DIMENSIONS
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        rank = U.shape[1]
        T = np.einsum('ijr,jkr,kr->ijjkik', U_3d, V_3d, W_3d)
        
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for kp in range(p):
                        for ip in range(n):
                            for k in range(p):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                if abs(T[i, j, jp, kp, ip, k] - expected) > 1e-9:
                                    return False
        return True

    # Try to find rank 11 solution
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    # Initial guess: Schoolbook rank 12, truncated to 11? No, random is better.
    # Or use a structured guess.
    
    best_solution = None
    best_residual = float('inf')
    
    # Try random restarts
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        np.random.seed(attempt)
        # Scale matters. Try small values.
        init_u = np.random.randn(dim_u, target_rank) * 0.5
        init_v = np.random.randn(dim_v, target_rank) * 0.5
        init_w = np.random.randn(dim_w, target_rank) * 0.5
        
        flat_init = np.concatenate([init_u.ravel(), init_v.ravel(), init_w.ravel()])
        
        try:
            res = least_squares(residual, flat_init, method='trf', max_nfev=200)
            if res.cost < best_residual:
                best_residual = res.cost
                best_solution = res.x
                
                # Check if solution is valid after snapping
                U_sol = best_solution[:dim_u * target_rank].reshape(dim_u, target_rank)
                V_sol = best_solution[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                W_sol = best_solution[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
                
                U_snap = snap_to_half_int(U_sol)
                V_snap = snap_to_half_int(V_sol)
                W_snap = snap_to_half_int(W_sol)
                
                if verify(U_snap, V_snap, W_snap):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
        except Exception as e:
            pass

    # If no rank 11 found, return schoolbook rank 12
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
