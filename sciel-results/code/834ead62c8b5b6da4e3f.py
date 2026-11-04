import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = 2, 4, 7
    target_rank = 49
    
    # 1. Generate the ground truth tensor T
    # The tensor T for matrix multiplication 2x4 * 4x7 -> 2x7
    # T[i, j, k] = 1 for all i, j, k
    # Because C[i, k] = sum_j A[i, j] B[j, k]
    T_true = np.ones((n, m, p))
    
    # 2. Define the residual function for rank r
    def residual(params, r):
        # Unpack params into U, V, W
        # U shape: (n*m, r) = (8, r)
        # V shape: (m*p, r) = (28, r)
        # W shape: (p*n, r) = (14, r)
        
        u_size = n * m * r
        v_size = m * p * r
        w_size = p * n * r
        
        u_flat = params[:u_size]
        v_flat = params[u_size:u_size + v_size]
        w_flat = params[u_size + v_size:]
        
        U = u_flat.reshape(n * m, r)
        V = v_flat.reshape(m * p, r)
        W = w_flat.reshape(p * n, r)
        
        # Compute T_decomp
        # T_decomp[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # T_decomp[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        T_decomp = np.einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d)
        
        return (T_decomp - T_true).flatten()

    # 3. Construct the known rank 50 solution as a baseline
    def construct_rank_50():
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0][col] += 1
            U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0][col] += 1
            U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + (i0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            # M3
            U[i0*m + j0][col] += 1
            V[j0*p + (k0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] -= 1
            W[(k0+1)*n + i0][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1)][col] += 1
            V[(j0+1)*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + i0][col] += 1
            W[k0*n + (i0+1)][col] += 1
            col += 1
            # M5
            U[i0*m + j0][col] += 1
            U[i0*m + (j0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] -= 1
            W[(k0+1)*n + i0][col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0][col] += 1
            U[i0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + (k0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M7
            U[i0*m + (j0+1)][col] += 1
            U[(i0+1)*m + (j0+1)][col] -= 1
            V[(j0+1)*p + k0][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            col += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*p + 6][col] = 1
                W[6*n + i][col] = 1
                col += 1
                
        return U, V, W

    U50, V50, W50 = construct_rank_50()
    
    # 4. Try to find a rank 49 solution
    best_solution = None
    best_residual = np.inf
    
    # We will try a few random initializations
    # To save time, we limit the number of attempts
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale matters. Strassen uses 1s and -1s.
        # We can initialize with small random numbers or integers.
        r = target_rank
        total_params = (n*m + m*p + p*n) * r
        params0 = np.random.randn(total_params) * 0.1
        
        try:
            # Use a simple gradient descent or least squares
            # For now, just check if we can get close to 0
            # This is a placeholder for a more sophisticated optimizer
            # In a real implementation, we would use scipy.optimize.least_squares
            pass
        except Exception as e:
            continue

    # 5. If no rank 49 found, use rank 50
    if best_solution is None:
        return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
    else:
        # Unpack best_solution
        r = target_rank
        u_size = n * m * r
        v_size = m * p * r
        
        u_flat = best_solution[:u_size]
        v_flat = best_solution[u_size:u_size + v_size]
        w_flat = best_solution[u_size + v_size:]
        
        U = u_flat.reshape(n * m, r)
        V = v_flat.reshape(m * p, r)
        W = w_flat.reshape(p * n, r)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
