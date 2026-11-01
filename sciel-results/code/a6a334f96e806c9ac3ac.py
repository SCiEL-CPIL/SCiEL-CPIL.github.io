import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook Rank
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    def compute_residuals(params, rank):
        len_U = n * m * rank
        len_V = m * p * rank
        U_flat = params[:len_U]
        V_flat = params[len_U:len_U+len_V]
        W_flat = params[len_U+len_V:]
        
        U = U_flat.reshape((n*m, rank))
        V = V_flat.reshape((m*p, rank))
        W = W_flat.reshape((p*n, rank))
        
        residuals = []
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]  # (m, rank)
                V_slice = V[0:m*p:1, :] # Wait, V_slice needs to be V[:, k] effectively?
                # V index is j*p + k.
                # We need V[j, k] for all j.
                # This is V[k::p] if row-major j*p+k?
                # If V is flattened row-major (j, k), index = j*p + k.
                # Then V[:, k] is V[k::p].
                
                # Let's extract V_col_k
                V_col_k = V[k::p].reshape(m, rank)
                
                w_val = W[k*n + i] # (rank,)
                
                # Term: sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
                # We want mat[j1, j2] = sum_r U[i, j1, r] * W[k, i, r] * V[j2, k, r]
                # mat = (U_slice * w_val[None, :]) @ V_col_k.T
                
                prod = U_slice * w_val[None, :]
                mat = prod @ V_col_k.T # (m, m)
                target = np.eye(m)
                residuals.append((mat - target).flatten())
        
        return np.concatenate(residuals)

    def check_exact(U, V, W, rank):
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        if not np.allclose(U2, np.round(U2), atol=1e-5):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-5):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-5):
            return False
        
        U_int = np.round(U2).astype(int)
        V_int = np.round(V2).astype(int)
        W_int = np.round(W2).astype(int)
        
        # Verify
        for i in range(n):
            for k in range(p):
                U_slice = U_int[i*m:(i+1)*m, :]
                V_col_k = V_int[k::p].reshape(m, rank)
                w_val = W_int[k*n + i]
                
                prod = U_slice * w_val[None, :]
                mat = prod @ V_col_k.T
                if not np.allclose(mat, 4 * np.eye(m), atol=1e-6):
                    return False
        return True

    # Try to reduce rank
    # Start from rank_schoolbook - 1 down to some limit
    # We will try a few ranks.
    
    for rank in range(rank_schoolbook - 1, 50, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        found = False
        # Try a few random restarts
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Initialize near schoolbook but with rank reduction?
            # Simple random init scaled
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            params_init = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(lambda x: compute_residuals(x, rank), params_init, method='lm', max_nfev=200)
                
                len_U = n * m * rank
                len_V = m * p * rank
                U_res = result.x[:len_U].reshape((n*m, rank))
                V_res = result.x[len_U:len_U+len_V].reshape((m*p, rank))
                W_res = result.x[len_U+len_V:].reshape((p*n, rank))
                
                if check_exact(U_res, V_res, W_res, rank):
                    best_rank = rank
                    best_U = U_res
                    best_V = V_res
                    best_W = W_res
                    found = True
                    break
            except Exception:
                pass
                
        if found:
            # If we found a lower rank, we can try even lower, but might take too long.
            # We'll continue to see if we can go lower.
            pass

    # Return best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
