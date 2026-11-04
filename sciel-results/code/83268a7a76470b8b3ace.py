"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = 120.0  # seconds
    
    def check_deadline():
        return time.time() - start_time > deadline - 5.0
    
    # Function to compute the error
    def compute_error(params, R):
        U = params[:R*n*m].reshape(R, n*m)
        V = params[R*n*m:R*n*m + R*m*p].reshape(R, m*p)
        W = params[R*n*m + R*m*p:].reshape(R, p*n)
        
        # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Vectorized:
        # U is (R, n*m), V is (R, m*p), W is (R, p*n)
        # We need to compute T[i,j,k] = sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
        
        # Expand dimensions for broadcasting
        # U_exp: (1, R, n, m)
        U_exp = U.T.reshape(n, m, R) # (n, m, R)
        V_exp = V.T.reshape(m, p, R) # (m, p, R)
        W_exp = W.T.reshape(p, n, R) # (p, n, R)
        
        # We want sum over R
        # U_exp[i,j,r] * V_exp[j,k,r] * W_exp[k,i,r]
        
        # This is a bit tricky to vectorize directly without huge intermediate arrays.
        # n=4, m=4, p=5, R~70.
        # U_exp: 4x4x70
        # V_exp: 4x5x70
        # W_exp: 5x4x70
        
        # Let's compute the tensor T of shape (n, m, p)
        T = np.zeros((n, m, p))
        for r in range(R):
            # U[:, :, r] is (n, m)
            # V[:, :, r] is (m, p)
            # W[:, :, r] is (p, n)
            # We need T[i,j,k] += U[i,j,r] * V[j,k,r] * W[k,i,r]
            # This is an element-wise product of three tensors with contracted indices?
            # No, it's a sum of rank-1 tensors.
            # T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i]
            
            # We can compute this by iterating over i, j, k?
            # Or use np.einsum
            # T += np.einsum('ij,jk,ki->ijk', U[:, :, r], V[:, :, r], W[:, :, r])
            T += np.einsum('ij,jk,ki->ijk', U[:, :, r], V[:, :, r], W[:, :, r])
            
        # The target is 1 for all i,j,k
        error = np.sum((T - 1.0)**2)
        return error

    def objective(params, R):
        if check_deadline():
            return 1e10
        return compute_error(params, R)

    best_rank = rank_schoolbook
    best_result = None
    
    # Try to find a decomposition with rank < rank_schoolbook
    # We'll try a few ranks
    for R in range(rank_schoolbook - 1, max(20, rank_schoolbook - 30), -1):
        if check_deadline():
            break
            
        # Random initialization
        np.random.seed(42 + R)
        params = np.random.randn(R * (n*m + m*p + p*n))
        
        # Use minimize
        res = minimize(
            lambda x: objective(x, R),
            params,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if res.success and res.fun < 1e-6:
            # Found a solution
            # Extract U, V, W
            U = res.x[:R*n*m].reshape(R, n*m).T
            V = res.x[R*n*m:R*n*m + R*m*p].reshape(R, m*p).T
            W = res.x[R*n*m + R*m*p:].reshape(R, p*n).T
            
            # Try to round to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Check if rounded solution is valid
            error_r = compute_error(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]), R)
            
            if error_r < 1e-6:
                best_rank = R
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                break # Found a good one, stop searching for lower ranks
            
    if best_result:
        return best_result
    else:
        # Fall back to schoolbook
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
