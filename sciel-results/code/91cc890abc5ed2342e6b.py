"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
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
    deadline = start_time + 115  # 5s safety margin
    
    # Precompute target tensor indices
    # T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    # We need: sum_c U[a,c] * V[b,c] * W[c_idx,c] = 1 for all valid (a,b,c)
    # where c_idx = k*n+i
    
    # Build list of (a, b, c_idx) tuples
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                tensor_indices.append((a, b, c_idx))
    
    def compute_residual(U, V, W):
        """Compute sum of squared errors for given U, V, W matrices."""
        total = 0.0
        for (a, b, c) in tensor_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            total += (val - 1.0) ** 2
        return total
    
    def residual_and_grad(params, rank):
        """Compute residual and gradient given flattened parameters."""
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        total = 0.0
        for (a, b, c) in tensor_indices:
            # val = sum_c U[a,c] * V[b,c] * W[c,c]
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            err = 2.0 * (val - 1.0)
            total += (val - 1.0) ** 2
            
            # Gradients
            U_grad[a, :] += err * V[b, :] * W[c, :]
            V_grad[b, :] += err * U[a, :] * W[c, :]
            W_grad[c, :] += err * U[a, :] * V[b, :]
        
        grad = np.concatenate([U_grad.flatten(), V_grad.flatten(), W_grad.flatten()])
        return total, grad
    
    def try_rank(rank, max_iterations=500, gtol=1e-12, max_restarts=10):
        """Try to find a decomposition of given rank using optimization."""
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                return None
            
            # Different initialization strategies
            if restart == 0:
                # Small random
                np.random.seed(42 + rank * 100 + restart)
                scale = 0.5
            elif restart == 1:
                # Larger random
                np.random.seed(123 + rank * 100 + restart)
                scale = 1.0
            elif restart == 2:
                # Sparse-like: mostly zeros with some ones
                np.random.seed(777 + rank * 100 + restart)
                scale = 0.0
            else:
                np.random.seed(random.randint(0, 100000))
                scale = 0.3 + 0.7 * (restart / max_restarts)
            
            U_init = np.random.randn(n*m, rank) * scale
            V_init = np.random.randn(m*p, rank) * scale
            W_init = np.random.randn(p*n, rank) * scale
            
            if scale == 0.0:
                # Sparse initialization
                for _ in range(rank * 2):
                    U_init[np.random.randint(n*m), np.random.randint(rank)] = 1.0
                for _ in range(rank * 2):
                    V_init[np.random.randint(m*p), np.random.randint(rank)] = 1.0
                for _ in range(rank * 2):
                    W_init[np.random.randint(p*n), np.random.randint(rank)] = 1.0
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use L-BFGS-B for better performance with many parameters
                result = minimize(
                    lambda x: residual_and_grad(x, rank)[0],
                    x0,
                    jac=lambda x: residual_and_grad(x, rank)[1],
                    method='L-BFGS-B',
                    options={'maxiter': max_iterations, 'gtol': gtol, 'maxls': 50}
                )
                
                if time.time() > deadline - 3:
                    return None
                
                final_residual = result.fun
                if final_residual < 1e-8:
                    # Found a valid decomposition!
                    U_flat = result.x[:n*m*rank]
                    V_flat = result.x[n*m*rank:n*m*rank + m*p*rank]
                    W_flat = result.x[n*m*rank + m*p*rank:]
                    
                    U = U_flat.reshape(n*m, rank)
                    V = V_flat.reshape(m*p, rank)
                    W = W_flat.reshape(p*n, rank)
                    
                    # Round to nearest half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify
                    valid = True
                    for (a, b, c) in tensor_indices:
                        val = np.dot(U_rounded[a, :], V_rounded[b, :] * W_rounded[c, :])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    
                    if valid:
                        U_list = [[float(x) for x in row] for row in U_rounded]
                        V_list = [[float(x) for x in row] for row in V_rounded]
                        W_list = [[float(x) for x in row] for row in W_rounded]
                        return {"U": U_list, "V": V_list, "W": W_list}
                    
                    # Try rounding to integers too
                    U_rounded_int = np.round(U)
                    V_rounded_int = np.round(V)
                    W_rounded_int = np.round(W)
                    
                    valid = True
                    for (a, b, c) in tensor_indices:
                        val = np.dot(U_rounded_int[a, :], V_rounded_int[b, :] * W_rounded_int[c, :])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    
                    if valid:
                        U_list = [[float(x) for x in row] for row in U_rounded_int]
                        V_list = [[float(x) for x in row] for row in V_rounded_int]
                        W_list = [[float(x) for x in row] for row in W_rounded_int]
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                continue
        
        return None
    
    # Also try random search with {-1, 0, 1} entries
    def try_random_search(rank, max_attempts=500):
        """Try random search with {-1, 0, 1} entries."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                return None
            
            U = np.random.choice([-1, 0, 1], size=(n*m, rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, rank))
            
            # Quick check using einsum
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W)
            
            # Check if matches target
            valid = True
            for (a, b, c) in tensor_indices:
                if abs(T_decomp[a, b, c] - 1.0) > 1e-10:
                    valid = False
                    break
            
            if valid:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    best_result = schoolbook()
    
    # Try ranks from low to high (we want the lowest rank)
    # Theoretical lower bound is around 20-25 for 3x3x4
    # Try a range of ranks
    for target_rank in range(20, 36):
        if time.time() > deadline - 5:
            break
        
        # First try random search (fast)
        result = try_random_search(target_rank, max_attempts=200)
        if result is not None:
            return result
        
        # Then try optimization
        result = try_rank(target_rank, max_iterations=300, gtol=1e-10, max_restarts=5)
        if result is not None:
            return result
    
    return best_result
# EVOLVE-BLOCK-END
