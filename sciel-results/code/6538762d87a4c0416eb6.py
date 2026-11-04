import numpy as np
import random

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor equation:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # for all i in [0,n), j in [0,m), k in [0,p)
    
    # Schoolbook solution (rank = n*m*p)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Function to verify a candidate solution
    def verify(U, V, W):
        # U, V, W are numpy arrays
        # Check if all entries are close to integer or half-integer
        # And if the tensor equation holds exactly (within float precision)
        
        # 1. Check integer/half-integer constraint
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if rounding changed values significantly
        if np.max(np.abs(U - U_r)) > 1e-6 or np.max(np.abs(V - V_r)) > 1e-6 or np.max(np.abs(W - W_r)) > 1e-6:
            return False, None, None, None
            
        # 2. Check tensor equation
        # We need to check for all i, j, k
        # sum_c U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c] == 1
        
        # We can do this efficiently by iterating or using einsum if shapes allowed, 
        # but indices are coupled.
        # Let's just loop, it's small (54 iterations)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Index in U
                    idx_u = i * m + j
                    # Index in V
                    idx_v = j * p + k
                    # Index in W
                    idx_w = k * n + i
                    
                    # Sum over c
                    val = np.dot(U_r[idx_u, :], V_r[idx_v, :] * W_r[idx_w, :])
                    if abs(val - 1.0) > 1e-6:
                        return False, None, None, None
                        
        return True, U_r, V_r, W_r

    # Try to find a lower rank solution using Gradient Descent
    # We try ranks from 40 down to 20
    # If found, return it.
    
    # Set a seed for reproducibility in the search, but vary for restarts
    np.random.seed(42)
    
    best_solution = None
    
    for r in range(40, 19, -1):
        # Try a few random restarts
        for restart in range(5):
            # Initialize U, V, W with small random values
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Gradient Descent
            lr = 0.01
            for step in range(1000):
                # Compute loss and gradients
                # Loss = sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2
                
                # We can compute the error term E[i,j,k]
                # grad_U[i,j,c] = 2 * E[i,j,k] * V[j,k,c] * W[k,i,c]
                # grad_V[j,k,c] = 2 * E[i,j,k] * U[i,j,c] * W[k,i,c]
                # grad_W[k,i,c] = 2 * E[i,j,k] * U[i,j,c] * V[j,k,c]
                
                # To avoid storing E for all i,j,k in memory explicitly in a loop 
                # that is too slow in Python, we can use vectorized operations 
                # or just accept the loop since n,m,p are small.
                
                # Let's use a loop for clarity and speed (54 iterations is fast)
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                total_loss = 0.0
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            
                            # Compute current value
                            # val = sum_c U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                            # Use dot product
                            val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                            err = val - 1.0
                            total_loss += err * err
                            
                            # Gradients
                            # d/dU_c = 2 * err * V_c * W_c
                            # d/dV_c = 2 * err * U_c * W_c
                            # d/dW_c = 2 * err * U_c * V_c
                            
                            grad_U[idx_u, :] += 2 * err * V[idx_v, :] * W[idx_w, :]
                            grad_V[idx_v, :] += 2 * err * U[idx_u, :] * W[idx_w, :]
                            grad_W[idx_w, :] += 2 * err * U[idx_u, :] * V[idx_v, :]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if total_loss < 1e-9:
                    break
            
            # Check if we found a solution
            is_valid, U_r, V_r, W_r = verify(U, V, W)
            if is_valid:
                best_solution = (U_r, V_r, W_r)
                break
        
        if best_solution is not None:
            break

    if best_solution is not None:
        U, V, W = best_solution
        # Convert to lists of lists for the return interface
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fall back to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
