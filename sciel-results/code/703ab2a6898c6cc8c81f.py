"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Try to find lower rank decompositions
    # The multiplication tensor has entries T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def tensor_residual(params, rank):
        """Compute residual given flattened params and rank."""
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :]) * 0  # This isn't right
                    # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    residual += (val - 1.0) ** 2
        return residual
    
    def tensor_residual_vec(params, rank):
        """Vectorized residual computation."""
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    total += (val - 1.0) ** 2
        return total
    
    def gradient(params, rank):
        """Compute gradient of residual."""
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = 2.0 * (val - 1.0)
                    # d/dU[i*m+j,c] = err * V[j*p+k,c] * W[k*n+i,c]
                    U_grad[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                    V_grad[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                    W_grad[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
        
        return np.concatenate([U_grad.flatten(), V_grad.flatten(), W_grad.flatten()])
    
    best_result = schoolbook()
    
    # Try ranks from 35 down to some minimum
    # For 3x3x4, let's try a few ranks
    for target_rank in [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]:
        if time.time() > deadline - 5:
            break
            
        rank = target_rank
        # Initialize with small random values
        np.random.seed(42 + rank)
        U_init = np.random.randn(n*m, rank) * 0.1
        V_init = np.random.randn(m*p, rank) * 0.1
        W_init = np.random.randn(p*n, rank) * 0.1
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = minimize(tensor_residual_vec, x0, args=(rank,), method='BFGS', 
                             jac=gradient, options={'maxiter': 1000, 'gtol': 1e-10})
            
            if time.time() > deadline - 5:
                break
                
            final_residual = result.fun
            if final_residual < 1e-8:
                # Found a valid decomposition! Round to nearest half-integers
                U_flat = result.x[:n*m*rank]
                V_flat = result.x[n*m*rank:n*m*rank + m*p*rank]
                W_flat = result.x[n*m*rank + m*p*rank:]
                
                U = U_flat.reshape(n*m, rank)
                V = V_flat.reshape(m*p, rank)
                W = W_flat.reshape(p*n, rank)
                
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to list of lists with integer/half-integer entries
                    U_list = [[float(x) for x in row] for row in U_rounded]
                    V_list = [[float(x) for x in row] for row in V_rounded]
                    W_list = [[float(x) for x in row] for row in W_rounded]
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
        except Exception as e:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
