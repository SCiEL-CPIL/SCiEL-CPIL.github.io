"""Matrix multiplication tensor decomposition for (2,3,10) using Strassen on (2,2,10) sub-block."""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Build the known rank-55 decomposition
    rank = 55
    U = np.zeros((n * m, rank))  # 6 rows
    V = np.zeros((m * p, rank))  # 30 rows
    W = np.zeros((p * n, rank))  # 20 rows
    
    col = 0
    
    # Strassen for (2,2,2) applied to each k-pair
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0, col] = 1
        U[4, col] = 1
        V[k0, col] = 1
        V[10+k1, col] = 1
        W[2*k0, col] = 1
        W[2*k1+1, col] = 1
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[3, col] = 1
        U[4, col] = 1
        V[k0, col] = 1
        W[2*k0+1, col] = 1
        W[2*k1+1, col] = -1
        col += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0, col] = 1
        V[k1, col] = 1
        V[10+k1, col] = -1
        W[2*k1, col] = 1
        W[2*k1+1, col] = 1
        col += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[4, col] = 1
        V[10+k0, col] = 1
        V[k0, col] = -1
        W[2*k0, col] = 1
        W[2*k0+1, col] = 1
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0, col] = 1
        U[1, col] = 1
        V[10+k1, col] = 1
        W[2*k0, col] = -1
        W[2*k1, col] = 1
        col += 1
        
        # M6 = (A[1,1]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[4, col] = 1
        U[0, col] = -1
        V[k0, col] = 1
        V[k1, col] = 1
        W[2*k1+1, col] = 1
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1, col] = 1
        U[4, col] = -1
        V[10+k0, col] = 1
        V[10+k1, col] = 1
        W[2*k0, col] = 1
        W[2*k1+1, col] = -1
        col += 1
    
    # Part 2: Schoolbook for j=2
    for i in range(2):
        for k in range(10):
            U[i*3+2, col] = 1
            V[20+k, col] = 1
            W[k*2+i, col] = 1
            col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Vectorized verification
    def verify(U_arr, V_arr, W_arr, n, m, p, max_checks=10):
        rank = U_arr.shape[1]
        random.seed(42)
        for _ in range(max_checks):
            A = np.random.uniform(-2, 2, (n, m))
            B = np.random.uniform(-2, 2, (m, p))
            C = A @ B
            # Compute via tensor decomposition
            # ua[i,j] = sum_r U[i*m+j, r] * A[i,j]  -- wait, this isn't right
            # The decomposition is: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k])
            # But that's not standard. Let me re-read the convention.
            # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
            # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # So C[i,k] = sum_{j} A[i,j]*B[j,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k])
            # Wait, that's not right either. Let me think again.
            # C[i,k] = sum_j A[i,j] B[j,k]
            # = sum_j sum_r T[i,j,k] A[i,j] B[j,k]  (since T=1)
            # = sum_r sum_j T[i,j,k] A[i,j] B[j,k]
            # = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k])
            # Hmm, that would require T[i,j,k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] summed over r
            # And C[i,k] = sum_j A[i,j]B[j,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k])
            # This is correct if the tensor decomposition is valid.
            
            # Compute: for each r, compute u_a[i,r] = sum_j U[i*m+j,r]*A[i,j] and v_b[k,r] = sum_j V[j*p+k,r]*B[j,k]
            # Then C[i,k] = sum_r W[k*n+i,r] * u_a[i,r] * v_b[k,r]
            
            # Vectorized:
            # u_a: shape (n, rank), u_a[i,r] = sum_j U[i*m+j,r]*A[i,j]
            U_reshaped = U_arr.reshape(n, m, rank)  # U_reshaped[i,j,r] = U[i*m+j, r]
            u_a = np.einsum('ijr,ij->ir', U_reshaped, A)  # u_a[i,r]
            
            V_reshaped = V_arr.reshape(m, p, rank)  # V_reshaped[j,k,r] = V[j*p+k, r]
            v_b = np.einsum('jkr,jk->kr', V_reshaped, B)  # v_b[k,r]
            
            W_reshaped = W_arr.reshape(p, n, rank)  # W_reshaped[k,i,r] = W[k*n+i, r]
            C_computed = np.einsum('kir,ir,kr->ik', W_reshaped, u_a, v_b)
            
            if np.max(np.abs(C - C_computed)) > 1e-9:
                return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find rank 54 by starting from rank-55 and eliminating one column
    best_solution = None
    
    search_start = time.time()
    max_search_time = 100.0  # seconds
    
    from scipy.optimize import least_squares
    
    # Strategy: start from rank-55, zero out one column, and adjust the rest
    # The residual is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    def make_residual_func(n, m, p, target_rank):
        def residuals(params):
            U_flat = params[:n*m*target_rank]
            V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank]
            W_flat = params[n*m*target_rank + m*p*target_rank:]
            
            U_mat = U_flat.reshape(n*m, target_rank)
            V_mat = V_flat.reshape(m*p, target_rank)
            W_mat = W_flat.reshape(p*n, target_rank)
            
            res = np.zeros(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[idx] = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :]) - 1.0
                        idx += 1
            return res
        return residuals
    
    # Try each column to eliminate
    for elim_col in range(rank):
        if time.time() - search_start > max_search_time:
            break
        if time.time() > DEADLINE - 20:
            break
        
        target_rank = rank - 1  # 54
        
        # Initialize: take rank-55 solution, remove column elim_col
        U_init = np.delete(U, elim_col, axis=1)
        V_init = np.delete(V, elim_col, axis=1)
        W_init = np.delete(W, elim_col, axis=1)
        
        # Add small perturbation to help escape local minima
        np.random.seed(elim_col)
        U_init = U_init + np.random.randn(*U_init.shape) * 0.01
        V_init = V_init + np.random.randn(*V_init.shape) * 0.01
        W_init = W_init + np.random.randn(*W_init.shape) * 0.01
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        residual_func = make_residual_func(n, m, p, target_rank)
        
        try:
            sol = least_squares(residual_func, params_init, method='lm', max_nfev=100)
            if sol.cost < 1e-12:
                U_sol = sol.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = sol.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = sol.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Round to nearest half-integer
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded, n, m, p):
                    best_solution = (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                    break
        except Exception:
            pass
    
    if best_solution:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    
    # If no better solution found, return the rank-55 solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
