import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook
    U = np.zeros((n*m, best_rank))
    V = np.zeros((m*p, best_rank))
    W = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank
    for r in range(best_rank - 1, 40, -1):
        if time.time() > deadline:
            break
            
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Random init
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Gradient descent
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline:
                    break
                    
                # Compute error
                # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # We need T[i,j,k] = 1 for all i,j,k
                # Error = sum (T - 1)^2
                
                # Efficient computation using einsum
                # U: (n*m, r), V: (m*p, r), W: (p*n, r)
                # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
                U_reshaped = U.reshape(n, m, r)
                V_reshaped = V.reshape(m, p, r)
                W_reshaped = W.reshape(p, n, r)
                
                # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # This is a bit tricky with einsum directly on the full tensor
                # T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
                # This might be slow for large r, but r is small.
                
                T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
                error = np.sum((T - 1)**2)
                
                if error < 1e-6:
                    break
                
                # Gradients
                # dT/dc = U[i,j,c] V[j,k,c] W[k,i,c]
                # dError/dU[i,j,c] = sum_{k} 2(T[i,j,k] - 1) V[j,k,c] W[k,i,c]
                # dError/dV[j,k,c] = sum_{i} 2(T[i,j,k] - 1) U[i,j,c] W[k,i,c]
                # dError/dW[k,i,c] = sum_{j} 2(T[i,j,k] - 1) U[i,j,c] V[j,k,c]
                
                grad_U = np.einsum('ijk,jkc,kic->ijc', (T - 1), V_reshaped, W_reshaped)
                grad_V = np.einsum('ijk,ijc,kic->jkc', (T - 1), U_reshaped, W_reshaped)
                grad_W = np.einsum('ijk,ijc,jkc->kic', (T - 1), U_reshaped, V_reshaped)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
            # Check if solution is close enough
            T = np.einsum('ijc,jkc,kic->ijk', U.reshape(n, m, r), V.reshape(m, p, r), W.reshape(p, n, r))
            if np.max(np.abs(T - 1)) < 1e-3:
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_rounded = np.einsum('ijc,jkc,kic->ijk', U_rounded.reshape(n, m, r), V_rounded.reshape(m, p, r), W_rounded.reshape(p, n, r))
                if np.max(np.abs(T_rounded - 1)) < 1e-6:
                    best_rank = r
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break # Found a solution for this rank, try lower
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
