"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R = 10  # Fixed bond dimension (rank) for tensor train decomposition
    
    def compute_tensor(U, V, W):
        """Compute T[i,j,k,l] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+l,r]"""
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T[i,j,k,l] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+l])
        return T
    
    def get_target_tensor():
        """Target tensor T[i,j,k,l] = delta_{jk}"""
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T[i,j,k,l] = 1.0 if j == k else 0.0
        return T
    
    def try_decomposition(seed):
        """Try to find a decomposition of rank R using randomized gradient descent."""
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, R) * 0.3
        V = np.random.randn(m * p, R) * 0.3
        W = np.random.randn(p * n, R) * 0.3
        
        T_target = get_target_tensor()
        
        # Adaptive learning rate gradient descent
        lr = 0.05
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for iteration in range(20000):
            T_current = compute_tensor(U, V, W)
            error = T_current - T_target
            loss = np.sum(error**2)
            
            if loss < 1e-12:
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
            
            # Momentum update
            vU = momentum * vU + (1 - momentum) * dU
            vV = momentum * vV + (1 - momentum) * dV
            vW = momentum * vW + (1 - momentum) * dW
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            # Decay learning rate
            if iteration % 5000 == 0 and iteration > 0:
                lr *= 0.5
        
        T_final = compute_tensor(U, V, W)
        max_err = np.max(np.abs(T_final - T_target))
        
        if max_err < 1e-8:
            return U, V, W
        return None, None, None
    
    # Try multiple seeds to find a good solution
    for seed in range(50):
        U, V, W = try_decomposition(seed)
        if U is not None:
            # Try to convert to integer or half-integer values
            # First, try scaling by various factors and rounding to integers
            best_solution = None
            best_err = float('inf')
            
            for scale in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]:
                scaled_U = U * scale
                scaled_V = V * scale
                scaled_W = W * scale
                
                # Try integer rounding
                U_int = np.round(scaled_U)
                V_int = np.round(scaled_V)
                W_int = np.round(scaled_W)
                
                max_round_err = max(
                    np.max(np.abs(U_int - scaled_U)),
                    np.max(np.abs(V_int - scaled_V)),
                    np.max(np.abs(W_int - scaled_W))
                )
                
                if max_round_err < 1e-6:
                    # Verify the integer solution
                    U_check = U_int.astype(float)
                    V_check = V_int.astype(float)
                    W_check = W_int.astype(float)
                    
                    T_check = compute_tensor(U_check, V_check, W_check)
                    T_target = get_target_tensor()
                    verify_err = np.max(np.abs(T_check - T_target))
                    
                    if verify_err < 1e-10:
                        U_list = [[int(x) for x in row] for row in U_check]
                        V_list = [[int(x) for x in row] for row in V_check]
                        W_list = [[int(x) for x in row] for row in W_check]
                        return {"U": U_list, "V": V_list, "W": W_list}
            
            # Try half-integer rounding (values that are multiples of 0.5)
            for scale in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]:
                scaled_U = U * scale
                scaled_V = V * scale
                scaled_W = W * scale
                
                # Round to nearest half-integer: round(2*x) / 2
                U_half = np.round(scaled_U * 2) / 2.0
                V_half = np.round(scaled_V * 2) / 2.0
                W_half = np.round(scaled_W * 2) / 2.0
                
                max_round_err = max(
                    np.max(np.abs(U_half - scaled_U)),
                    np.max(np.abs(V_half - scaled_V)),
                    np.max(np.abs(W_half - scaled_W))
                )
                
                if max_round_err < 1e-6:
                    # Verify the half-integer solution
                    U_check = U_half
                    V_check = V_half
                    W_check = W_half
                    
                    T_check = compute_tensor(U_check, V_check, W_check)
                    T_target = get_target_tensor()
                    verify_err = np.max(np.abs(T_check - T_target))
                    
                    if verify_err < 1e-10:
                        U_list = [[float(x) for x in row] for row in U_check]
                        V_list = [[float(x) for x in row] for row in V_check]
                        W_list = [[float(x) for x in row] for row in W_check]
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook decomposition
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
