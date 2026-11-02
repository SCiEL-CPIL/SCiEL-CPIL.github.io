import numpy as np
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # We search for U (n, m, r), V (m, p, r), W (p, n, r)
    # such that sum_c U[i, j, c] V[j, k, c] W[k, i, c] == 1
    
    def local_search(r, max_iters=50000, temp_init=1.0, cool_rate=0.99):
        # Initialize U, V, W with random integers in {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(n, m, r))
        V = np.random.randint(-1, 2, size=(m, p, r))
        W = np.random.randint(-1, 2, size=(p, n, r))
        
        # Calculate initial S and error
        S = np.zeros((n, m, p))
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        S[i, j, k] += U[i, j, c] * V[j, k, c] * W[k, i, c]
                        
        error = np.sum((1.0 - S)**2)
        
        if error < 1e-9:
            return U, V, W
            
        current_temp = temp_init
        for it in range(max_iters):
            var_type = np.random.randint(3)
            
            if var_type == 0:
                i, j, c = np.random.randint(n), np.random.randint(m), np.random.randint(r)
                old_val = U[i, j, c]
                new_val = np.random.randint(-1, 2)
                if new_val == old_val: continue
                
                dU = new_val - old_val
                delta_error = 0
                for k in range(p):
                    delta_S = dU * V[j, k, c] * W[k, i, c]
                    old_term = 1.0 - S[i, j, k]
                    new_term = old_term - delta_S
                    delta_error += new_term**2 - old_term**2
                    
                if delta_error < 0 or (current_temp > 0 and random.random() < np.exp(-delta_error / current_temp)):
                    U[i, j, c] = new_val
                    for k in range(p):
                        S[i, j, k] += dU * V[j, k, c] * W[k, i, c]
                    error += delta_error
            elif var_type == 1:
                j, k, c = np.random.randint(m), np.random.randint(p), np.random.randint(r)
                old_val = V[j, k, c]
                new_val = np.random.randint(-1, 2)
                if new_val == old_val: continue
                
                dV = new_val - old_val
                delta_error = 0
                for i in range(n):
                    delta_S = U[i, j, c] * dV * W[k, i, c]
                    old_term = 1.0 - S[i, j, k]
                    new_term = old_term - delta_S
                    delta_error += new_term**2 - old_term**2
                    
                if delta_error < 0 or (current_temp > 0 and random.random() < np.exp(-delta_error / current_temp)):
                    V[j, k, c] = new_val
                    for i in range(n):
                        S[i, j, k] += U[i, j, c] * dV * W[k, i, c]
                    error += delta_error
            else:
                k, i, c = np.random.randint(p), np.random.randint(n), np.random.randint(r)
                old_val = W[k, i, c]
                new_val = np.random.randint(-1, 2)
                if new_val == old_val: continue
                
                dW = new_val - old_val
                delta_error = 0
                for j in range(m):
                    delta_S = U[i, j, c] * V[j, k, c] * dW
                    old_term = 1.0 - S[i, j, k]
                    new_term = old_term - delta_S
                    delta_error += new_term**2 - old_term**2
                    
                if delta_error < 0 or (current_temp > 0 and random.random() < np.exp(-delta_error / current_temp)):
                    W[k, i, c] = new_val
                    for j in range(m):
                        S[i, j, k] += U[i, j, c] * V[j, k, c] * dW
                    error += delta_error
            
            current_temp *= cool_rate
            if error < 1e-9:
                return U, V, W
                
        return None

    # Try ranks from 17 down to 1
    for r in range(17, 0, -1):
        for attempt in range(5):
            result = local_search(r, max_iters=20000)
            if result is not None:
                U, V, W = result
                U_flat = U.reshape(n * m, r).tolist()
                V_flat = V.reshape(m * p, r).tolist()
                W_flat = W.reshape(p * n, r).tolist()
                return {"U": U_flat, "V": V_flat, "W": W_flat}
        
    # Fallback to schoolbook rank 18
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
