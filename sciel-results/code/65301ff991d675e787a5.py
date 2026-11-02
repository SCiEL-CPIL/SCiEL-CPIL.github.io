"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
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
    
    def to_half_integer_array(arr):
        rounded = np.round(arr * 2) / 2
        return [[float(x) for x in row] for row in rounded]
    
    def check_decomposition(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    start_time = time.time()
    deadline = start_time + 110
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from 24 to 35
    for target_rank in range(24, rank_schoolbook):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Use gradient descent with a penalty for non-half-integer values
            lr = 0.001
            penalty_weight = 1.0
            
            for step in range(5000):
                if time.time() > deadline:
                    break
                
                # Compute error and gradient
                err = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            d_err = 2 * (val - 1.0)
                            err += (val - 1.0) ** 2
                            
                            for c in range(target_rank):
                                u_val = U[i*m+j, c]
                                v_val = V[j*p+k, c]
                                w_val = W[k*n+i, c]
                                grad_U[i*m+j, c] += d_err * v_val * w_val
                                grad_V[j*p+k, c] += d_err * u_val * w_val
                                grad_W[k*n+i, c] += d_err * u_val * v_val
                
                # Add penalty for non-half-integer values
                for arr, grad_arr in [(U, grad_U), (V, grad_V), (W, grad_W)]:
                    # Penalty: sum of (2*x - round(2*x))^2
                    two_x = 2 * arr
                    rounded = np.round(two_x)
                    penalty_grad = penalty_weight * 4 * (two_x - rounded)
                    grad_arr += penalty_grad
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if err < 1e-12:
                    break
            
            # Round to half-integers and check
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if check_decomposition(U_r, V_r, W_r, tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if best_rank == target_rank:
            break
    
    if best_rank == rank_schoolbook:
        return {"U": to_half_integer_array(U_sb), 
                "V": to_half_integer_array(V_sb), 
                "W": to_half_integer_array(W_sb)}
    else:
        return {"U": to_half_integer_array(best_U), 
                "V": to_half_integer_array(best_V), 
                "W": to_half_integer_array(best_W)}
# EVOLVE-BLOCK-END
