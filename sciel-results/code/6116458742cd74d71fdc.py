"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Schoolbook rank = n * m * p = 4 * 4 * 5 = 80
    schoolbook_rank = n * m * p
    
    # Create the schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = schoolbook_rank
    
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Try to find a lower-rank decomposition using border rank heuristic
    # with gradient descent and rational reconstruction
    
    def compute_residual(U, V, W):
        """Compute the residual tensor: 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        residual = np.ones((n, m, p))
        for r in range(U.shape[1]):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i, j, k] -= U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return residual
    
    def frobenius_norm_sq(residual):
        """Compute squared Frobenius norm of residual tensor"""
        return np.sum(residual ** 2)
    
    def gradient_descent(U_init, V_init, W_init, max_iter=2000, lr=0.01, tol=1e-12):
        """Perform gradient descent on the squared Frobenius norm of the tensor residual"""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        rank = U.shape[1]
        
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute residual
            residual = np.ones((n, m, p))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual[i, j, k] -= U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            loss = frobenius_norm_sq(residual)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < tol:
                break
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # d/dU[i*m+j,r] of sum (1 - U*V*W)^2 = -2 * residual * V * W
                            grad_U[i*m+j, r] += -2.0 * residual[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += -2.0 * residual[i, j, k] * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += -2.0 * residual[i, j, k] * U[i*m+j, r] * V[j*p+k, r]
            
            # Update with adaptive learning rate
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            if it % 100 == 99:
                lr *= 0.9
            
            # Periodically check if we've converged to a good solution
            if it % 50 == 0 and loss < 1e-6:
                break
        
        return best_U, best_V, best_W, best_loss
    
    def rational_reconstruct(U, V, W, rank, tol=1e-6):
        """Try to snap the factors to half-integers if they're close"""
        U_r = np.zeros_like(U)
        V_r = np.zeros_like(V)
        W_r = np.zeros_like(W)
        
        for i in range(U.shape[0]):
            for j in range(rank):
                val = U[i, j]
                nearest = round(val * 2) / 2
                if abs(val - nearest) < tol:
                    U_r[i, j] = nearest
                else:
                    U_r[i, j] = val
        
        for i in range(V.shape[0]):
            for j in range(rank):
                val = V[i, j]
                nearest = round(val * 2) / 2
                if abs(val - nearest) < tol:
                    V_r[i, j] = nearest
                else:
                    V_r[i, j] = val
        
        for i in range(W.shape[0]):
            for j in range(rank):
                val = W[i, j]
                nearest = round(val * 2) / 2
                if abs(val - nearest) < tol:
                    W_r[i, j] = nearest
                else:
                    W_r[i, j] = val
        
        # Verify the reconstructed solution
        residual = compute_residual(U_r, V_r, W_r)
        if frobenius_norm_sq(residual) < 1e-8:
            return U_r, V_r, W_r, True
        return U, V, W, False
    
    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that the decomposition is exact"""
        residual = compute_residual(U, V, W)
        return frobenius_norm_sq(residual) < tol
    
    # Try different ranks from low to high
    for target_rank in range(40, min(schoolbook_rank, 80)):
        if time.time() > deadline - 10:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Multiple random initializations
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            # Random complex-like initialization (use real values with random signs)
            np.random.seed(42 + attempt + target_rank)
            U_init = np.random.randn(n * m, target_rank) * 0.1
            V_init = np.random.randn(m * p, target_rank) * 0.1
            W_init = np.random.randn(p * n, target_rank) * 0.1
            
            # Run gradient descent
            U_opt, V_opt, W_opt, loss = gradient_descent(U_init, V_init, W_opt, max_iter=500, lr=0.005)
            
            # Check if we found a good solution
            if loss < 1e-8:
                # Try rational reconstruction
                U_r, V_r, W_r, success = rational_reconstruct(U_opt, V_opt, W_opt, target_rank)
                
                if success and verify_decomposition(U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    break  # Found a valid solution at this rank, try lower
                elif verify_decomposition(U_opt, V_opt, W_opt):
                    # Use the non-rational solution if it's valid
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = target_rank
                    break
    
    # Convert to lists with integer/half-integer entries where possible
    def to_list_with_ints(arr):
        result = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if abs(val - round(val)) < 1e-8:
                    row.append(int(round(val)))
                elif abs(val - (round(val*2)/2)) < 1e-8:
                    row.append(round(val*2)/2)
                else:
                    row.append(float(val))
            result.append(row)
        return result
    
    U_list = to_list_with_ints(best_U)
    V_list = to_list_with_ints(best_V)
    W_list = to_list_with_ints(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
