"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a rank-8 decomposition
    # The tensor T[i,j,k,l] = delta_{jk} for i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r] = delta_{jk}
    
    # Use a known construction or solve numerically
    # Let's try rank 8 first
    
    def try_rank(R):
        """Try to find a decomposition of rank R using random initialization and gradient descent."""
        np.random.seed(42)
        
        # Initialize with small random values
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        
        # Compute the tensor from current factors
        def compute_tensor(U, V, W):
            """Compute T[i,j,k,l] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+l,r]"""
            T = np.zeros((n, m, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(m):
                        for l in range(p):
                            T[i,j,k,l] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+l])
            return T
        
        # Target tensor
        T_target = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T_target[i,j,k,l] = 1.0 if j == k else 0.0
        
        # Gradient descent
        lr = 0.01
        for iteration in range(5000):
            T_current = compute_tensor(U, V, W)
            error = T_current - T_target
            loss = np.sum(error**2)
            
            if loss < 1e-10:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(m):
                        for l in range(p):
                            e = error[i,j,k,l]
                            for r in range(R):
                                dU[i*m+j, r] += 2 * e * V[j*p+k, r] * W[k*n+l, r]
                                dV[j*p+k, r] += 2 * e * U[i*m+j, r] * W[k*n+l, r]
                                dW[k*n+l, r] += 2 * e * U[i*m+j, r] * V[j*p+k, r]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        T_final = compute_tensor(U, V, W)
        max_err = np.max(np.abs(T_final - T_target))
        
        if max_err < 1e-6:
            return U, V, W, max_err
        return None, None, None, max_err
    
    # Try ranks from 8 upward
    for R in range(8, 13):
        U, V, W, err = try_rank(R)
        if U is not None:
            # Convert to exact rational/integer form
            # Scale to make values integers or half-integers
            # Find a common scaling factor
            all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Try to find a scale factor that makes all values close to integers or half-integers
            best_scale = None
            best_max_err = float('inf')
            
            for scale in [1, 2, 4, 8, 16, 32, 64, 128]:
                scaled_U = U * scale
                scaled_V = V * scale
                scaled_W = W * scale
                
                # Check how close to integers
                U_int = np.round(scaled_U)
                V_int = np.round(scaled_V)
                W_int = np.round(scaled_W)
                
                U_err = np.max(np.abs(U_int - scaled_U))
                V_err = np.max(np.abs(V_int - scaled_V))
                W_err = np.max(np.abs(W_int - scaled_W))
                
                max_err_scaled = max(U_err, V_err, W_err)
                if max_err_scaled < 1e-4 and max_err_scaled < best_max_err:
                    best_max_err = max_err_scaled
                    best_scale = scale
                    best_U_int = U_int
                    best_V_int = V_int
                    best_W_int = W_int
            
            if best_scale is not None:
                # Verify the integer solution
                U_final = best_U_int.astype(float)
                V_final = best_V_int.astype(float)
                W_final = best_W_int.astype(float)
                
                # Check if this is an exact solution
                T_check = np.zeros((n, m, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(m):
                            for l in range(p):
                                T_check[i,j,k,l] = np.sum(U_final[i*m+j] * V_final[j*p+k] * W_final[k*n+l])
                
                T_target_check = np.zeros((n, m, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(m):
                            for l in range(p):
                                T_target_check[i,j,k,l] = 1.0 if j == k else 0.0
                
                if np.max(np.abs(T_check - T_target_check)) < 1e-8:
                    # Return as lists of integers
                    U_list = [[int(x) for x in row] for row in U_final]
                    V_list = [[int(x) for x in row] for row in V_final]
                    W_list = [[int(x) for x in row] for row in W_final]
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # If integer rounding didn't work, try half-integers
            for scale in [2, 4, 8, 16]:
                scaled_U = U * scale
                scaled_V = V * scale
                scaled_W = W * scale
                
                # Check closeness to half-integers (i.e., 2*x is close to integer)
                U_half = np.round(scaled_U * 2) / 2
                V_half = np.round(scaled_V * 2) / 2
                W_half = np.round(scaled_W * 2) / 2
                
                U_err = np.max(np.abs(U_half - scaled_U))
                V_err = np.max(np.abs(V_half - scaled_V))
                W_err = np.max(np.abs(W_half - scaled_W))
                
                max_err_half = max(U_err, V_err, W_err)
                if max_err_half < 1e-4:
                    U_final = U_half
                    V_final = V_half
                    W_final = W_half
                    
                    T_check = np.zeros((n, m, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(m):
                                for l in range(p):
                                    T_check[i,j,k,l] = np.sum(U_final[i*m+j] * V_final[j*p+k] * W_final[k*n+l])
                    
                    T_target_check = np.zeros((n, m, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(m):
                                for l in range(p):
                                    T_target_check[i,j,k,l] = 1.0 if j == k else 0.0
                    
                    if np.max(np.abs(T_check - T_target_check)) < 1e-8:
                        # Convert to half-integers represented as fractions with denominator 2
                        # But the interface wants integer/half-integer entries
                        # Let's represent them as-is (they are already half-integers)
                        U_list = [[float(x) for x in row] for row in U_final]
                        V_list = [[float(x) for x in row] for row in V_final]
                        W_list = [[float(x) for x in row] for row in W_final]
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
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
