import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline to be safe within 150s limit
    
    # Precompute indices for U, V, W updates
    u_indices = []
    for i in range(n):
        for j in range(m):
            I = i * m + j
            v_rows = np.arange(j * p, (j + 1) * p)
            w_rows = np.array([k * n + i for k in range(p)])
            u_indices.append((v_rows, w_rows))
            
    v_indices = []
    for j in range(m):
        for k in range(p):
            J = j * p + k
            u_rows = np.array([i * m + j for i in range(n)])
            w_rows = np.array([k * n + i for i in range(n)])
            v_indices.append((u_rows, w_rows))
            
    w_indices = []
    for k in range(p):
        for i in range(n):
            K = k * n + i
            u_rows = np.array([i * m + j for j in range(m)])
            v_rows = np.array([j * p + k for j in range(m)])
            w_indices.append((u_rows, v_rows))

    best_solution = None
    best_rank = n * m * p
    
    # Fallback solution (Schoolbook)
    U_fb = np.zeros((n * m, best_rank))
    V_fb = np.zeros((m * p, best_rank))
    W_fb = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
                
    # Try ranks from 25 to 50
    for R in range(25, 51):
        if time.time() > deadline:
            break
            
        # Multiple restarts for each rank
        max_restarts = 15
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(12345 + R * 1000 + restart * 7)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Scale to help convergence
            U *= 0.5
            V *= 0.5
            W *= 0.5
            
            best_error = float('inf')
            best_U, best_V, best_W = None, None, None
            
            # ALS Iterations
            max_iters = 300
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                for I in range(n * m):
                    v_rows, w_rows = u_indices[I]
                    B = V[v_rows, :] * W[w_rows, :]
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                    U[I, :] = x
                    
                # Update V
                for J in range(m * p):
                    u_rows, w_rows = v_indices[J]
                    B = U[u_rows, :] * W[w_rows, :]
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[J, :] = x
                    
                # Update W
                for K in range(p * n):
                    u_rows, v_rows = w_indices[K]
                    B = U[u_rows, :] * V[v_rows, :]
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    W[K, :] = x
                
                # Calculate Error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        I = i * m + j
                        for k in range(p):
                            J = j * p + k
                            K = k * n + i
                            val = np.dot(U[I, :], V[J, :] * W[K, :])
                            err += (val - 1.0) ** 2
                
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if best_error < 1e-10:
                    break
            
            if best_error < 1e-8:
                # Round to half-integers
                U_int = np.round(best_U * 2) / 2
                V_int = np.round(best_V * 2) / 2
                W_int = np.round(best_W * 2) / 2
                
                # Verify Exactness
                is_exact = True
                for i in range(n):
                    for j in range(m):
                        I = i * m + j
                        for k in range(p):
                            J = j * p + k
                            K = k * n + i
                            val = np.dot(U_int[I, :], V_int[J, :] * W_int[K, :])
                            if abs(val - 1.0) > 1e-9:
                                is_exact = False
                                break
                        if not is_exact: break
                    if not is_exact: break
                    
                if is_exact:
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
        
    # Fallback
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
# EVOLVE-BLOCK-END
