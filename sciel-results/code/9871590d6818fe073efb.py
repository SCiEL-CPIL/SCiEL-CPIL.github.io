"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook rank as fallback
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

    # Build the multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need to find rank-r decomposition:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U is (n*m, r), V is (m*p, r), W is (p*n, r)
    
    # Reshape for easier handling
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Try small ranks with random initialization and gradient descent
    # The objective is: minimize sum_{i,j,k} (T[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    best_loss = 0.0

    # Try ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But we can try smaller
    min_rank = max(n*m, m*p, p*n)
    
    # Actually, for matrix multiplication, the tensor rank is known to be less than n*m*p
    # For 3x4x5, let's try ranks from 20 upward
    
    # Use a more systematic approach: try specific ranks with multiple random restarts
    ranks_to_try = list(range(min_rank, min_rank + 15))
    
    # For each rank, try several random initializations
    np.random.seed(42)
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        max_restarts = 50 if rank <= 25 else 20
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Scale to help convergence
            scale = (n * m * p) ** (1.0 / 6.0)
            U *= scale
            V *= scale
            W *= scale
            
            # Gradient descent with adaptive learning rate
            lr = 0.01
            best_iter_loss = float('inf')
            stagnant = 0
            
            for it in range(500):
                if time.time() > deadline:
                    break
                
                # Compute current tensor
                # T_approx[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # Loss = sum_{i,j,k} (1 - T_approx[i,j,k])^2
                
                # Compute gradients
                # dLoss/dU[i*m+j, c] = -2 * sum_{k} (1 - T_approx[i,j,k]) * V[j*p+k, c] * W[k*n+i, c]
                # dLoss/dV[j*p+k, c] = -2 * sum_{i} (1 - T_approx[i,j,k]) * U[i*m+j, c] * W[k*n+i, c]
                # dLoss/dW[k*n+i, c] = -2 * sum_{j} (1 - T_approx[i,j,k]) * U[i*m+j, c] * V[j*p+k, c]
                
                # Efficient computation using einsum
                # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
                
                # Reshape for einsum
                U_reshaped = U.reshape(n, m, rank)
                V_reshaped = V.reshape(m, p, rank)
                W_reshaped = W.reshape(p, n, rank)
                
                # T_approx[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                T_approx = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
                
                # Residual
                R = 1.0 - T_approx  # shape (n, m, p)
                
                loss = np.sum(R ** 2)
                
                if loss < best_iter_loss:
                    best_iter_loss = loss
                    stagnant = 0
                else:
                    stagnant += 1
                
                if stagnant > 50:
                    break
                
                # Check for exact solution
                if abs(loss) < 1e-10:
                    found = True
                    break
                
                # Compute gradients
                # dU[i,j,c] = -2 * sum_k R[i,j,k] * V[j,k,c] * W[k,i,c]
                dU = -2.0 * np.einsum('ijk,jkc,kic->ijc', R, V_reshaped, W_reshaped)
                # dV[j,k,c] = -2 * sum_i R[i,j,k] * U[i,j,c] * W[k,i,c]
                dV = -2.0 * np.einsum('ijk,ijc,kic->jkc', R, U_reshaped, W_reshaped)
                # dW[k,i,c] = -2 * sum_j R[i,j,k] * U[i,j,c] * V[j,k,c]
                dW = -2.0 * np.einsum('ijk,ijc,jkc->kic', R, U_reshaped, V_reshaped)
                
                # Apply gradients
                U -= lr * dU.reshape(n*m, rank)
                V -= lr * dV.reshape(m*p, rank)
                W -= lr * dW.reshape(p*n, rank)
                
                # Adaptive learning rate
                if loss < best_iter_loss * 0.99:
                    lr = min(lr * 1.05, 0.1)
                elif loss > best_iter_loss:
                    lr *= 0.8
                
                best_iter_loss = min(best_iter_loss, loss)
            
            if found:
                # We found an exact decomposition
                # Verify and return
                U_reshaped = U.reshape(n, m, rank)
                V_reshaped = V.reshape(m, p, rank)
                W_reshaped = W.reshape(p, n, rank)
                T_approx = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
                if np.max(np.abs(T_approx - 1.0)) < 1e-8:
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    best_loss = 0.0
                    break
        
        if found:
            break
    
    # Convert to lists for the return interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
