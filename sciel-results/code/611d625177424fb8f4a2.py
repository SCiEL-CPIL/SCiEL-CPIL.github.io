import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    rank_schoolbook = n * m * p  # 36

    # Build schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def verify(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Verify schoolbook
    assert verify(U_school, V_school, W_school), "Schoolbook verification failed"

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook

    # Strategy: Try to find lower-rank decompositions using optimization
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is equivalent to finding the tensor rank of the all-ones tensor
    # with specific index structure.
    
    # Approach: Use a structured parameterization and optimize
    # Let's try different rank values and use gradient-based optimization
    
    def try_rank(R, max_iter=2000, lr=0.01, seed=42):
        """Try to find a rank-R decomposition using optimization."""
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # Also try structured initialization
        U_struct = np.zeros((n * m, R))
        V_struct = np.zeros((m * p, R))
        W_struct = np.zeros((p * n, R))
        
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    U_struct[i*m+j, r] = 1.0
            for j in range(m):
                for k in range(p):
                    V_struct[j*p+k, r] = 1.0
            for k in range(p):
                for i in range(n):
                    W_struct[k*n+i, r] = 1.0
        
        # Scale to get sum = 1
        U_struct /= R
        V_struct /= R
        W_struct /= R
        
        # Try both initializations
        for init_name, U0, V0, W0 in [("random", U, V, W), ("structured", U_struct, V_struct, W_struct)]:
            U_curr = U0.copy()
            V_curr = V0.copy()
            W_curr = W0.copy()
            
            # Gradient descent on the loss: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                    
                # Compute predictions
                pred = np.zeros((n, m, p))
                for r in range(R):
                    pred += np.outer(np.outer(
                        U_curr[:, r].reshape(n, m),
                        V_curr[:, r].reshape(m, p)
                    ).T, W_curr[:, r].reshape(p, n).T).T
                    # Actually, let's be more careful with the indexing
                    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This is a 3D tensor with indices (i, j, k)
                    u_r = U_curr[:, r].reshape(n, m)
                    v_r = V_curr[:, r].reshape(m, p)
                    w_r = W_curr[:, r].reshape(p, n)
                    # pred[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                pred[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
                
                # Compute loss
                loss = np.sum((pred - 1.0)**2)
                
                if loss < 1e-12:
                    # Found exact solution
                    return U_curr, V_curr, W_curr, True
                
                # Compute gradients (simplified - using finite differences would be too slow)
                # Instead, use a simple coordinate descent approach
                
                # Update U
                for r in range(R):
                    for i in range(n):
                        for j in range(m):
                            # d/dU[i*m+j,r] of loss
                            grad = 0
                            for k in range(p):
                                val = pred[i,j,k] - 1.0
                                grad += 2 * val * V_curr[j*p+k, r] * W_curr[k*n+i, r]
                            U_curr[i*m+j, r] -= lr * grad
                
                # Update V
                for r in range(R):
                    for j in range(m):
                        for k in range(p):
                            grad = 0
                            for i in range(n):
                                val = pred[i,j,k] - 1.0
                                grad += 2 * val * U_curr[i*m+j, r] * W_curr[k*n+i, r]
                            V_curr[j*p+k, r] -= lr * grad
                
                # Update W
                for r in range(R):
                    for k in range(p):
                        for i in range(n):
                            grad = 0
                            for j in range(m):
                                val = pred[i,j,k] - 1.0
                                grad += 2 * val * U_curr[i*m+j, r] * V_curr[j*p+k, r]
                            W_curr[k*n+i, r] -= lr * grad
                
                if it % 100 == 0:
                    # Check if we're making progress
                    pass
        
        # Check if we found a valid solution
        for U_final, V_final, W_final in [(U, V, W), (U_struct, V_struct, W_struct)]:
            if verify(U_final, V_final, W_final, tol=1e-8):
                return U_final, V_final, W_final, True
        
        return None, None, None, False

    # Try ranks from 35 down to 25
    for R in range(rank_schoolbook - 1, 24, -1):
        if time.time() > deadline - 10:
            break
            
        result_U, result_V, result_W, success = try_rank(R, max_iter=500, lr=0.05, seed=R)
        
        if success:
            best_U, best_V, best_W = result_U, result_V, result_W
            best_rank = R
            break  # Found the lowest rank we can achieve with this approach
    
    # If we didn't find a better decomposition, return schoolbook
    if best_rank == rank_schoolbook:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Verify the best solution
    if verify(best_U, best_V, best_W, tol=1e-8):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
