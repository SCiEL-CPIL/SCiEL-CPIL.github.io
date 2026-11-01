"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 120.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
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
    
    # Initialize best with schoolbook
    best_U, best_V, best_W = schoolbook()
    
    # Define residual calculation
    def compute_residuals(U, V, W, rank):
        """
        Residuals: for all i, k, j1, j2:
        sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] - delta(j1, j2)
        """
        residuals = []
        for i in range(n):
            for k in range(p):
                # Matrix of size m x m
                # Entry (j1, j2) is sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r]
                # U_slice: (m, rank)
                # V_slice: (m, rank) -> transpose to (rank, m)
                # W_scalar: scalar
                U_slice = U[i*m:(i+1)*m, :]  # (m, rank)
                V_slice = V[k*p:(k+1)*p, :]  # (m, rank)
                w_val = W[k*n + i]
                
                # Term: sum_r U_slice[j1, r] * V_slice[j2, r] * w_val[r]
                # = (U_slice * w_val[None, :]) @ V_slice.T
                # (m, rank) @ (rank, m) -> (m, m)
                prod = U_slice * w_val[None, :]
                mat = prod @ V_slice.T
                target = np.eye(m)
                residuals.append(mat.flatten() - target.flatten())
        
        return np.concatenate(residuals)
    
    def check_exact(U, V, W, rank):
        """
        Check if U, V, W are integers or half-integers and satisfy the identity.
        """
        # Scale by 2 to check for integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        # Check if they are close to integers
        if not np.allclose(U2, np.round(U2), atol=1e-5):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-5):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-5):
            return False
        
        # Convert to exact integers
        U_int = np.round(U2).astype(int)
        V_int = np.round(V2).astype(int)
        W_int = np.round(W2).astype(int)
        
        # Verify identity with integers: sum_r U2 V2 W2 = 4 * delta
        # Since we scaled by 2, the identity is sum_r (U_int/2)(V_int/2)(W_int/2) = delta
        # => sum_r U_int V_int W_int = 4 * delta
        for i in range(n):
            for k in range(p):
                U_slice = U_int[i*m:(i+1)*m, :]
                V_slice = V_int[k*p:(k+1)*p, :]
                w_val = W_int[k*n + i]
                
                prod = U_slice * w_val[None, :]
                mat = prod @ V_slice.T
                if not np.allclose(mat, 4 * np.eye(m), atol=1e-6):
                    return False
        return True
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound for M(n,m,p) is at least max(nm, mp, pn) = max(10, 30, 12) = 30
    # Also known: M(n,m,p) >= max(n,m,p) * max(n,m,p) ... no, simple bounds.
    # Let's try from 30 to best_rank - 1
    
    # Random initialization strategy
    # We can use random normal distributions
    
    # To improve search, we can fix some gauge freedoms.
    # e.g., Fix first row of W to be 1s? No, that might be too restrictive.
    # Instead, let's just run random restarts.
    
    max_restarts_per_rank = 100
    time_limit_per_rank = 10.0
    
    # We will iterate ranks in increasing order
    # Start from 30
    min_rank = 30
    current_rank = min_rank
    
    while current_rank < best_rank:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        rank_time_start = time.time()
        restarts = 0
        found = False
        
        while restarts < max_restarts_per_rank:
            if time.time() - start_time > DEADLINE - 5:
                break
            if time.time() - rank_time_start > time_limit_per_rank:
                break
            
            # Random init
            U = np.random.randn(n * m, current_rank)
            V = np.random.randn(m * p, current_rank)
            W = np.random.randn(p * n, current_rank)
            
            # Simple iterative refinement (Gradient Descent or Least Squares)
            # Since we don't have scipy import guaranteed to be fast/available in all contexts? 
            # "Python standard library, NumPy and SciPy are available."
            # Let's try using scipy.optimize.least_squares if possible, else simple GD.
            # GD is slow but safe. Let's try a few steps of GD.
            # Or even better, just check if random init happens to be close? Unlikely.
            
            # Let's use scipy if available for speed
            try:
                from scipy.optimize import least_squares
                
                def fun(params):
                    # params: concatenated U, V, W
                    len_U = n * m * current_rank
                    len_V = m * p * current_rank
                    len_W = p * n * current_rank
                    
                    U_flat = params[:len_U]
                    V_flat = params[len_U:len_U+len_V]
                    W_flat = params[len_U+len_V:]
                    
                    U = U_flat.reshape((n*m, current_rank))
                    V = V_flat.reshape((m*p, current_rank))
                    W = W_flat.reshape((p*n, current_rank))
                    
                    return compute_residuals(U, V, W, current_rank)
                
                params_init = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                # Run least squares
                result = least_squares(fun, params_init, method='lm', max_nfev=200)
                
                # Extract result
                len_U = n * m * current_rank
                len_V = m * p * current_rank
                len_W = p * n * current_rank
                
                U_res = result.x[:len_U].reshape((n*m, current_rank))
                V_res = result.x[len_U:len_U+len_V].reshape((m*p, current_rank))
                W_res = result.x[len_U+len_V:].reshape((p*n, current_rank))
                
                # Check for integer/half-integer solution
                if check_exact(U_res, V_res, W_res, current_rank):
                    best_rank = current_rank
                    best_U = U_res
                    best_V = V_res
                    best_W = W_res
                    found = True
                    break
                    
            except Exception:
                pass
            
            restarts += 1
            
        if found:
            # Move to next rank? No, we found a valid one, we want minimal.
            # Since we are iterating upwards, the first found is minimal.
            # But wait, could there be a lower rank? We started at 30.
            # If 30 works, return. If not, try 31.
            # If we found one at current_rank, it's the minimum possible if we start from lower bound.
            # However, the lower bound might be higher than 30.
            # If we find one at 30, we are done.
            # If we don't, we continue to 31.
            # So if found, we can return immediately.
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        current_rank += 1
        
    # If no better rank found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
