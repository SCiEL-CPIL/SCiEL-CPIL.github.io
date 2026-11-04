import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    # Target tensor T
    # Indices: I = i*m + j, J = j*p + k, K = k*n + i
    # T[I, J, K] = 1 if consistent, else 0
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    T = np.zeros((size_u, size_v, size_w))
    
    # Fill T
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
                
    # Try to find a lower rank decomposition
    # Known rank for 3x5x6 is 52
    candidate_ranks = [52, 53, 54, 55, 56, 58, 60, 65, 70, 75, 80, 85, 89]
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    for rank in candidate_ranks:
        if time.time() > deadline - 5.0:
            break
            
        # Try a few restarts for this rank
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize U, V, W randomly
            U = np.random.randn(size_u, rank) * 0.1
            V = np.random.randn(size_v, rank) * 0.1
            W = np.random.randn(size_w, rank) * 0.1
            
            # Gradient Descent
            lr = 0.01
            prev_loss = float('inf')
            
            for step in range(200):
                if time.time() > deadline - 5.0:
                    break
                    
                # Compute X
                # X[I, J, K] = sum_r U[I, r] V[J, r] W[K, r]
                # Using einsum: 'ir,jr,kr->ijk'
                X = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
                
                # Loss
                diff = X - T
                loss = np.sum(diff**2)
                
                # Check for convergence
                if loss < 1e-6:
                    break
                
                # If loss increases, reduce lr
                if loss > prev_loss:
                    lr *= 0.5
                prev_loss = loss
                
                # Compute Gradients
                # dU[I, r] = sum_{J,K} diff[I, J, K] V[J, r] W[K, r]
                dU = np.einsum('ijk,jr,kr->ir', diff, V, W, optimize=True)
                dV = np.einsum('ijk,ir,kr->jr', diff, U, W, optimize=True)
                dW = np.einsum('ijk,ir,jr->kr', diff, U, V, optimize=True)
                
                # Update
                U -= 2 * lr * dU
                V -= 2 * lr * dV
                W -= 2 * lr * dW
                
            # After optimization, try to round to integers
            # First try integer rounding
            U_int = np.rint(U)
            V_int = np.rint(V)
            W_int = np.rint(W)
            
            X_int = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int, optimize=True)
            if np.allclose(X_int, T, atol=1e-9):
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
                
            # Try half-integer rounding (multiply by 2, round, divide by 2)
            U_half = np.rint(U * 2) / 2.0
            V_half = np.rint(V * 2) / 2.0
            W_half = np.rint(W * 2) / 2.0
            
            X_half = np.einsum('ir,jr,kr->ijk', U_half, V_half, W_half, optimize=True)
            if np.allclose(X_half, T, atol=1e-9):
                return {
                    "U": U_half.tolist(),
                    "V": V_half.tolist(),
                    "W": W_half.tolist()
                }
                
            # If not converged or rounded correctly, try next restart
            
    # Fallback to schoolbook
    rank = rank_school
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
