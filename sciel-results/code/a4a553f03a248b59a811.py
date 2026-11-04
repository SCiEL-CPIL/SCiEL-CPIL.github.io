"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # Target tensor for matrix multiplication: T[i,j,k] = 1
    target = np.ones((n, m, p))
    
    # Define objective function using vectorized operations
    def make_objective(rank):
        def objective(params):
            # Unpack parameters
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank:n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            
            # Reshape to 3D tensors
            # U: (n, m, rank)
            # V: (m, p, rank)
            # W: (p, n, rank)
            U_3d = U_flat.reshape(n, m, rank)
            V_3d = V_flat.reshape(m, p, rank)
            W_3d = W_flat.reshape(p, n, rank)
            
            # Compute tensor product sum
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            
            # Frobenius norm of error
            error = np.sum((T - target) ** 2)
            return error
        return objective

    # Try ranks starting from a lower bound
    # For 3x3x3, rank is 23. For 3x3x5, we expect it to be somewhat higher or similar.
    # Start from 23 to be safe, though 3x3x5 might need more.
    for rank in range(23, n * m * p + 1):
        # Multiple restarts to escape local minima
        for restart in range(10):
            np.random.seed(42 + rank * 100 + restart)
            
            # Initialize with small random values
            # Using uniform distribution for better Nelder-Mead behavior
            U = np.random.uniform(-0.5, 0.5, (n*m, rank))
            V = np.random.uniform(-0.5, 0.5, (m*p, rank))
            W = np.random.uniform(-0.5, 0.5, (p*n, rank))
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                objective = make_objective(rank)
                result = minimize(objective, params0, method='Nelder-Mead', 
                                  options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
                
                if result.fun < 1e-8:
                    # Extract optimized parameters
                    U_opt = result.x[:n*m*rank].reshape(n, m, rank)
                    V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m, p, rank)
                    W_opt = result.x[n*m*rank + m*p*rank:].reshape(p, n, rank)
                    
                    # Round to nearest half-integer
                    U_r = np.round(U_opt * 2) / 2
                    V_r = np.round(V_opt * 2) / 2
                    W_r = np.round(W_opt * 2) / 2
                    
                    # Verify exactness
                    T_check = np.einsum('ijr,jkr,kir->ijk', 
                                        U_r, 
                                        V_r, 
                                        W_r)
                    
                    if np.allclose(T_check, target, atol=1e-10):
                        # Return in the required format (list of lists)
                        # U: (n*m, rank)
                        # V: (m*p, rank)
                        # W: (p*n, rank)
                        return {
                            "U": U_r.flatten().reshape(n*m, rank).tolist(),
                            "V": V_r.flatten().reshape(m*p, rank).tolist(),
                            "W": W_r.flatten().reshape(p*n, rank).tolist()
                        }
            except Exception:
                pass

    # Fallback: Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
