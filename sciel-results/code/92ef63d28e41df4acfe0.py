import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook decomposition
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Target tensor T[i, j, k] = 1 if i, j, k are valid indices for the multiplication
    # T is the tensor of the bilinear map A, B -> C
    # C[i, k] = sum_j A[i, j] B[j, k]
    # So T[i, j, k] = 1 for all i, j, k
    # Wait, the tensor T has indices i, j, k.
    # T[i, j, k] = 1.
    # The decomposition is sum_c U[i, c] V[j, c] W[k, c] = T[i, j, k]
    # So T is the all-ones tensor of shape (n, m, p).
    
    T = np.ones((n, m, p))
    
    def check_decomposition(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = np.array(U).reshape(n, m, rank)
        V_reshaped = np.array(V).reshape(m, p, rank)
        W_reshaped = np.array(W).reshape(p, n, rank)
        
        # Compute the tensor product sum
        # T_calc[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        # Note: W index is k*n + i, so W_reshaped[k, i, c]
        
        T_calc = np.zeros((n, m, p))
        for c in range(rank):
            # u_c is (n, m), v_c is (m, p), w_c is (p, n)
            # We need to compute the sum of outer products?
            # No, T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            # This is not a simple outer product.
            # It is a trilinear form.
            # Let's iterate over i, j, k
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_calc[i, j, k] += U_reshaped[i, j, c] * V_reshaped[j, k, c] * W_reshaped[k, i, c]
                        
        return np.allclose(T_calc, T, atol=1e-9)

    def round_to_half_integer(X):
        return np.round(np.array(X) * 2) / 2.0

    # Try to find a lower rank decomposition
    # Start from rank 22
    for target_rank in [22, 23]:
        if time.time() > deadline:
            break
            
        # Initialize with random values
        # We can try to initialize with the schoolbook decomposition truncated?
        # Or random.
        
        # Let's try a simple local search
        # We define a cost function
        def cost(params):
            U_flat, V_flat, W_flat = params[:n*m*target_rank], params[n*m*target_rank:n*m*target_rank + m*p*target_rank], params[n*m*target_rank + m*p*target_rank:]
            U = U_flat.reshape(n, m, target_rank)
            V = V_flat.reshape(m, p, target_rank)
            W = W_flat.reshape(p, n, target_rank)
            
            T_calc = np.zeros((n, m, p))
            for c in range(target_rank):
                # This loop is slow. Use vectorization if possible.
                # T[i, j, k] += U[i, j, c] V[j, k, c] W[k, i, c]
                # We can use einsum or similar, but the indices are mixed.
                # Let's just use the loop for now, it's small.
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_calc[i, j, k] += U[i, j, c] * V[j, k, c] * W[k, i, c]
                            
            return np.sum((T_calc - T)**2)

        # Use scipy.optimize.minimize
        from scipy.optimize import minimize
        
        x0 = np.random.randn(n*m*target_rank + m*p*target_rank + p*n*target_rank)
        
        best_cost = float('inf')
        best_x = None
        
        for restart in range(10): # 10 restarts
            if time.time() > deadline:
                break
            x0 = np.random.randn(n*m*target_rank + m*p*target_rank + p*n*target_rank)
            res = minimize(cost, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x
                
            if best_cost < 1e-6:
                break
                
        if best_cost < 1e-6:
            U_flat, V_flat, W_flat = best_x[:n*m*target_rank], best_x[n*m*target_rank:n*m*target_rank + m*p*target_rank], best_x[n*m*target_rank + m*p*target_rank:]
            U = round_to_half_integer(U_flat.reshape(n, m, target_rank))
            V = round_to_half_integer(V_flat.reshape(m, p, target_rank))
            W = round_to_half_integer(W_flat.reshape(p, n, target_rank))
            
            # Flatten to 2D
            U_2d = U.reshape(n*m, target_rank)
            V_2d = V.reshape(m*p, target_rank)
            W_2d = W.reshape(p*n, target_rank)
            
            # Convert to list of lists
            U_list = U_2d.tolist()
            V_list = V_2d.tolist()
            W_list = W_2d.tolist()
            
            if check_decomposition(U_list, V_list, W_list, target_rank):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
