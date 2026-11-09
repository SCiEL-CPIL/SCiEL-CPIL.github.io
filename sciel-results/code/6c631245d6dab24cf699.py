import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    timeout = 110.0
    
    n, m, p = 3, 3, 5
    size_U = n * m  # 9
    size_V = m * p  # 15
    size_W = p * n  # 15
    
    # Build the trivial rank-45 decomposition
    R0 = n * m * p  # 45
    U0 = np.zeros((size_U, R0))
    V0 = np.zeros((size_V, R0))
    W0 = np.zeros((size_W, R0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1.0
                V0[j*p+k, col] = 1.0
                W0[k*n+i, col] = 1.0
                col += 1
    
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = R0
    
    def verify(U, V, W, R):
        """Verify that U, V, W with rank R give exact decomposition."""
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i,j,:], V_r[j,k,:] * W_r[k,i,:])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_halfint(U, V, W, R):
        """Check if entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for v in row:
                    if abs(v * 2 - round(v * 2)) > 1e-9:
                        return False
        return True
    
    # Try to find rank 44, 43, 42 by removing one term and adjusting
    # Strategy: for each rank R from 44 down to 40, try to find a valid decomposition
    # by starting from the naive one and perturbing
    
    for R in range(R0 - 1, 39, -1):
        if time.time() - start_time > timeout - 15:
            break
        
        found = False
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() - start_time > timeout - 10:
                break
            
            np.random.seed(12345 + attempt * 7)
            
            # Start from a perturbation of the naive solution
            # Take first R columns of the naive solution and add small perturbation
            if R <= R0:
                U = U0[:, :R].copy()
                V = V0[:, :R].copy()
                W = W0[:, :R].copy()
                
                # Add small random perturbation to break symmetry
                scale = 0.05
                U += scale * np.random.randn(size_U, R)
                V += scale * np.random.randn(size_V, R)
                W += scale * np.random.randn(size_W, R)
            else:
                U = np.random.randn(size_U, R) * 0.1
                V = np.random.randn(size_V, R) * 0.1
                W = np.random.randn(size_W, R) * 0.1
            
            # Use gradient descent to minimize the residual
            lr = 0.01
            for it in range(200):
                if time.time() - start_time > timeout - 10:
                    break
                
                # Compute gradient
                U_r = U.reshape(n, m, R)
                V_r = V.reshape(m, p, R)
                W_r = W.reshape(p, n, R)
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                            # dT/dU[i,j,r] = V[j,k,r] * W[k,i,r]
                            # dT/dV[j,k,r] = U[i,j,r] * W[k,i,r]
                            # dT/dW[k,i,r] = U[i,j,r] * V[j,k,r]
                            residual = 1.0 - np.dot(U_r[i,j,:], V_r[j,k,:] * W_r[k,i,:])
                            dU[i*m+j, :] += residual * V_r[j,k,:] * W_r[k,i,:]
                            dV[j*p+k, :] += residual * U_r[i,j,:] * W_r[k,i,:]
                            dW[k*n+i, :] += residual * U_r[i,j,:] * V_r[j,k,:]
                
                # Gradient descent step
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                U_r = U.reshape(n, m, R)
                V_r = V.reshape(m, p, R)
                W_r = W.reshape(p, n, R)
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_r[i,j,:], V_r[j,k,:] * W_r[k,i,:])
                            max_res = max(max_res, abs(val - 1.0))
                
                if max_res < 1e-8:
                    break
            
            # Check if we found a valid solution
            if verify(U, V, W, R):
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify(U_r, V_r, W_r, R) and verify_halfint(U_r, V_r, W_r, R):
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = R
                    found = True
                    break
        
        if found:
            break
    
    return best_result
