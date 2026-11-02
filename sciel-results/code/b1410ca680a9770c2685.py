import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook solution as fallback
    def schoolbook():
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
    
    U_sb, V_sb, W_sb = schoolbook()
    
    def to_half_int(arr):
        """Round to nearest half-integer."""
        return np.round(arr * 2) / 2.0

    def check_exact(U, V, W, T, tol=1e-9):
        """Check if sum_c U[:,c] \otimes V[:,c] \otimes W[:,c] == T"""
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(recon, T, atol=tol)
        
    def residuals(params, U_shape, V_shape, W_shape):
        """Flatten params to U, V, W and compute residual tensor."""
        U_size = U_shape[0] * U_shape[1]
        V_size = V_shape[0] * V_shape[1]
        W_size = W_shape[0] * W_shape[1]
        
        U = params[:U_size].reshape(U_shape)
        V = params[U_size:U_size+V_size].reshape(V_shape)
        W = params[U_size+V_size:U_size+V_size+W_size].reshape(W_shape)
        
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return (recon - T).flatten()
        
    def try_rank(r, max_restarts=10, seed=42):
        """Try to find a rank-r decomposition using least_squares."""
        np.random.seed(seed)
        
        U_shape = (n * m, r)
        V_shape = (m * p, r)
        W_shape = (p * n, r)
        
        U_size = U_shape[0] * U_shape[1]
        V_size = V_shape[0] * V_shape[1]
        W_size = W_shape[0] * W_shape[1]
        total_size = U_size + V_size + W_size
        
        best_err = np.inf
        best_params = None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            scale = 1.0 / np.sqrt(r)
            params0 = np.random.randn(total_size) * scale
            
            try:
                result = least_squares(
                    residuals, 
                    params0, 
                    args=(U_shape, V_shape, W_shape),
                    method='lm',
                    max_nfev=1000
                )
                if result.cost < best_err:
                    best_err = result.cost
                    best_params = result.x
                    if best_err < 1e-10:
                        break
            except Exception:
                try:
                    result = least_squares(
                        residuals, 
                        params0, 
                        args=(U_shape, V_shape, W_shape),
                        method='trf',
                        max_nfev=1000
                    )
                    if result.cost < best_err:
                        best_err = result.cost
                        best_params = result.x
                        if best_err < 1e-10:
                            break
                except Exception:
                    pass
        
        if best_params is None:
            return None
            
        if best_err > 1e-6:
            return None
            
        U = best_params[:U_size].reshape(U_shape)
        V = best_params[U_size:U_size+V_size].reshape(V_shape)
        W = best_params[U_size+V_size:U_size+V_size+W_size].reshape(W_shape)
        
        U_h = to_half_int(U)
        V_h = to_half_int(V)
        W_h = to_half_int(W)
        
        if check_exact(U_h, V_h, W_h, T):
            return U_h, V_h, W_h
            
        return None

    # Compute matricization ranks to get a lower bound
    T1 = T.reshape(n*m, m*p*p*n)
    T2 = T.transpose(1,0,2).reshape(m*p, n*m*p*n)
    T3 = T.transpose(2,0,1).reshape(p*n, n*m*m*p)
    
    rank1 = np.linalg.matrix_rank(T1)
    rank2 = np.linalg.matrix_rank(T2)
    rank3 = np.linalg.matrix_rank(T3)
    
    min_rank_bound = max(rank1, rank2, rank3)
    
    best_result = None
    # Try ranks from lower bound up to schoolbook
    for r in range(min_rank_bound, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
            
        restarts = 20 if r < 30 else 10
        
        found = try_rank(r, max_restarts=restarts, seed=r)
        if found is not None:
            U, V, W = found
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback: schoolbook
    U_list = [[float(x) for x in row] for row in U_sb]
    V_list = [[float(x) for x in row] for row in V_sb]
    W_list = [[float(x) for x in row] for row in W_sb]
    return {"U": U_list, "V": V_list, "W": W_list}
