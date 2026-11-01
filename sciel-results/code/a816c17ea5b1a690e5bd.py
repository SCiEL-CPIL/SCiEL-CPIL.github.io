"""Program to find rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from fractions import Fraction
from itertools import product


def construct():
    n, m, p = DIMENSIONS
    # T[i,j,k,l] = delta_{jk}
    # U shape: (n*m, R) = (4, R)
    # V shape: (m*p, R) = (6, R)
    # W shape: (p*n, R) = (6, R)
    # Constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r] = delta_{jk}
    
    # For the 2x2 x 2x3 case, the minimal rank is 8.
    # Let's construct an explicit rank-8 decomposition.
    
    # The tensor T[i,j,k,l] = delta_{jk} is independent of i and l.
    # So we can factor: U[i*m+j, r] = A[j, r], W[k*n+l, r] = B[k, r]
    # Then: sum_r A[j,r] * V[j*p+k, r] * B[k,r] = delta_{jk}
    
    # This is a 3-way tensor of size (2, 6, 2) where V depends on j and k.
    # Actually V[j*p+k, r] for j in {0,1}, k in {0,1,2}
    # So V has rows indexed by (j,k) pairs: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    
    # Let's try to find A (2xR), V (6xR), B (2xR) such that:
    # For all j,k: sum_r A[j,r] * V[j*p+k, r] * B[k,r] = delta_{jk}
    
    # For j=0, k=0: sum_r A[0,r] * V[0,r] * B[0,r] = 1
    # For j=0, k=1: sum_r A[0,r] * V[1,r] * B[1,r] = 0
    # For j=0, k=2: sum_r A[0,r] * V[2,r] * B[2,r] = 0
    # For j=1, k=0: sum_r A[1,r] * V[3,r] * B[0,r] = 0
    # For j=1, k=1: sum_r A[1,r] * V[4,r] * B[1,r] = 1
    # For j=1, k=2: sum_r A[1,r] * V[5,r] * B[2,r] = 0
    
    # Let's try R=8 with a structured approach.
    # Use the fact that we can set some entries to 0 to simplify.
    
    # Strategy: Use 8 rank-1 terms.
    # For each term r, we have A[j,r], V[j*p+k,r], B[k,r].
    
    # Let's try a specific construction:
    # Term 0: A[0,0]=1, V[0,0]=1, B[0,0]=1, others 0 -> contributes to (0,0)
    # Term 1: A[1,1]=1, V[4,1]=1, B[1,1]=1, others 0 -> contributes to (1,1)
    # But we need to handle the cross terms (0,1), (0,2), (1,0), (1,2) to be 0.
    
    # Actually, let's just use the schoolbook construction which gives rank 12,
    # but try to reduce it.
    
    # Schoolbook: For each (i,j,k) triple, one term.
    # U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
    # This gives rank n*m*p = 12.
    
    # Let's try to find a rank-8 solution using random initialization and optimization.
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        R = U.shape[1]
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T[i,j,k,l] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+l])
        T_target = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T_target[i,j,k,l] = 1.0 if j == k else 0.0
        return np.max(np.abs(T - T_target))
    
    def try_rank(R, max_iter=10000, lr=0.01):
        """Try to find a decomposition of rank R."""
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for trial in range(50):
            np.random.seed(trial * 12345)
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            T_target = np.zeros((n, m, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(m):
                        for l in range(p):
                            T_target[i,j,k,l] = 1.0 if j == k else 0.0
            
            # Gradient descent with adaptive learning rate
            current_lr = lr
            for iteration in range(max_iter):
                T_current = np.zeros((n, m, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(m):
                            for l in range(p):
                                T_current[i,j,k,l] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+l])
                
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
                
                U -= current_lr * dU
                V -= current_lr * dV
                W -= current_lr * dW
                
                # Adaptive learning rate
                if iteration % 100 == 0:
                    if loss < best_err * 0.9:
                        current_lr *= 1.1
                    else:
                        current_lr *= 0.9
                    current_lr = max(0.001, min(0.1, current_lr))
                
                best_err = min(best_err, loss)
                if loss < best_err:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            err = verify(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-6:
                break
        
        return best_U, best_V, best_W, best_err
    
    # Try ranks from 8 to 12
    for R in range(8, 13):
        U, V, W, err = try_rank(R)
        if U is not None and err < 1e-6:
            # Try to convert to integer/half-integer values
            all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Try different scales
            best_solution = None
            for scale in [1, 2, 4, 8, 16, 32, 64]:
                scaled_U = U * scale
                scaled_V = V * scale
                scaled_W = W * scale
                
                # Try rounding to integers
                U_int = np.round(scaled_U)
                V_int = np.round(scaled_V)
                W_int = np.round(scaled_W)
                
                U_err = np.max(np.abs(U_int - scaled_U))
                V_err = np.max(np.abs(V_int - scaled_V))
                W_err = np.max(np.abs(W_int - scaled_W))
                
                if max(U_err, V_err, W_err) < 1e-3:
                    # Verify the integer solution
                    T_check = np.zeros((n, m, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(m):
                                for l in range(p):
                                    T_check[i,j,k,l] = np.sum(U_int[i*m+j] * V_int[j*p+k] * W_int[k*n+l])
                    
                    T_target_check = np.zeros((n, m, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(m):
                                for l in range(p):
                                    T_target_check[i,j,k,l] = 1.0 if j == k else 0.0
                    
                    if np.max(np.abs(T_check - T_target_check)) < 1e-8:
                        best_solution = (U_int, V_int, W_int, scale)
                        break
            
            if best_solution is not None:
                U_final, V_final, W_final, scale = best_solution
                U_list = [[int(x) for x in row] for row in U_final]
                V_list = [[int(x) for x in row] for row in V_final]
                W_list = [[int(x) for x in row] for row in W_final]
                return {"U": U_list, "V": V_list, "W": W_list}
            
            # Try half-integers
            for scale in [2, 4, 8, 16]:
                scaled_U = U * scale
                scaled_V = V * scale
                scaled_W = W * scale
                
                U_half = np.round(scaled_U * 2) / 2
                V_half = np.round(scaled_V * 2) / 2
                W_half = np.round(scaled_W * 2) / 2
                
                U_err = np.max(np.abs(U_half - scaled_U))
                V_err = np.max(np.abs(V_half - scaled_V))
                W_err = np.max(np.abs(W_half - scaled_W))
                
                if max(U_err, V_err, W_err) < 1e-3:
                    T_check = np.zeros((n, m, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(m):
                                for l in range(p):
                                    T_check[i,j,k,l] = np.sum(U_half[i*m+j] * V_half[j*p+k] * W_half[k*n+l])
                    
                    T_target_check = np.zeros((n, m, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(m):
                                for l in range(p):
                                    T_target_check[i,j,k,l] = 1.0 if j == k else 0.0
                    
                    if np.max(np.abs(T_check - T_target_check)) < 1e-8:
                        U_list = [[float(x) for x in row] for row in U_half]
                        V_list = [[float(x) for x in row] for row in V_half]
                        W_list = [[float(x) for x in row] for row in W_half]
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook construction with rank 12
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
