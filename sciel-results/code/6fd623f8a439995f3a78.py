import numpy as np
import time

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    # Target tensor: T[i, j, k] = 1 for all i, j, k
    # Equation: sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We try ranks from low to high.
    # Lower bound: max(n*m, m*p, p*n) = 18.
    # But we know Strassen gives 32. Let's try to beat 32.
    # Start from 18 up to 31.
    
    best_result = None
    best_rank = None
    
    # Fallback: Trivial rank 36
    rank_trivial = n * m * p
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1.0
                V_trivial[j*p+k, col] = 1.0
                W_trivial[k*n+i, col] = 1.0
                col += 1
    
    for R in range(18, 36):
        if time.time() > deadline:
            break
            
        # Multiple restarts for each rank to escape local minima
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            np.random.seed(42 + restart + R)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # ALS Iterations
            converged = False
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Update W
                # A[i, k, r] = sum_j U[i, j, r] * V[j, k, r]
                # Reshape U to (n, m, R), V to (m, p, R)
                U_reshaped = U.reshape(n, m, R)
                V_reshaped = V.reshape(m, p, R)
                # A[i, k, r] = sum_j U[i, j, r] V[j, k, r]
                # This is a batched matmul: U (n, m, R) * V^T (m, p, R) -> (n, p, R)
                # V^T has shape (p, m, R)? No.
                # V is (m, p, R). We want to multiply U (n, m, R) with V (m, p, R) along axis 1 (m).
                # np.einsum('ijr,jkr->ikr', U_reshaped, V_reshaped)
                A = np.einsum('ijr,jkr->ikr', U_reshaped, V_reshaped)
                
                # System: sum_r A[i, k, r] * W[k*n+i, r] = 1
                # W is (p*n, R). Index is k*n+i.
                # We need to solve for W.
                # For each (i, k), we have a linear equation in W[k*n+i, :].
                # A[i, k, :] @ W[k*n+i, :] = 1
                # This is a system of n*p equations, each with R unknowns.
                # We can stack them.
                # A_flat: (n*p, R)
                # W_flat: (n*p, R)
                # We want A_flat @ W_flat.T = ones(n*p)?
                # No, W is stored as (p*n, R).
                # Let's create a matrix M of shape (n*p, R) where M[idx, r] = W[idx, r]
                # And A_flat of shape (n*p, R) where A_flat[idx, r] = A[i, k, r]
                # Then sum_r M[idx, r] * A_flat[idx, r] = 1.
                # This is M @ A_flat.T = ones? No.
                # It's element-wise sum: np.sum(M * A_flat, axis=1) = 1.
                # This is not a standard linear system Mx=b because M is the variable.
                # Wait, W is the variable.
                # Equation: sum_r W[idx, r] * A[idx, r] = 1.
                # This is a linear equation in W[idx, :].
                # Since the equations for different (i,k) are decoupled (each involves only W[k*n+i, :]),
                # we can solve each one independently?
                # No, W has R components for each (i,k).
                # So for each (i,k), we have: A[i,k,:] . W[k*n+i, :] = 1.
                # This is an underdetermined system (R unknowns, 1 equation).
                # We can find the minimum norm solution:
                # W[k*n+i, :] = A[i,k,:] / (A[i,k,:] . A[i,k,:])
                # But we also need to ensure consistency if we were solving a system.
                # Since it's just one equation per (i,k), the min-norm solution is valid.
                
                W_new = np.zeros_like(W)
                for i in range(n):
                    for k in range(p):
                        idx = k * n + i
                        a_vec = A[i, k, :]
                        norm_sq = np.dot(a_vec, a_vec)
                        if norm_sq > 1e-10:
                            W_new[idx, :] = a_vec / norm_sq
                        else:
                            W_new[idx, :] = 0
                W = W_new
                
                # Update U
                # B[i, j, r] = sum_k V[j, k, r] * W[k, i, r]
                # V_reshaped: (m, p, R)
                # W_reshaped: (p, n, R)
                B = np.einsum('jkr,kir->ijr', V_reshaped, W.reshape(p, n, R))
                
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        idx = i * m + j
                        b_vec = B[i, j, :]
                        norm_sq = np.dot(b_vec, b_vec)
                        if norm_sq > 1e-10:
                            U_new[idx, :] = b_vec / norm_sq
                        else:
                            U_new[idx, :] = 0
                U = U_new
                
                # Update V
                # C[j, k, r] = sum_i U[i, j, r] * W[k, i, r]
                C = np.einsum('ijr,kir->jkr', U_reshaped, W.reshape(p, n, R))
                
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        c_vec = C[j, k, :]
                        norm_sq = np.dot(c_vec, c_vec)
                        if norm_sq > 1e-10:
                            V_new[idx, :] = c_vec / norm_sq
                        else:
                            V_new[idx, :] = 0
                V = V_new
                
                # Check residual
                # Residual = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] - 1
                U_reshaped = U.reshape(n, m, R)
                V_reshaped = V.reshape(m, p, R)
                W_reshaped = W.reshape(p, n, R)
                Prod = np.einsum('ijr,jkr,kir->ik', U_reshaped, V_reshaped, W_reshaped)
                Residual = np.max(np.abs(Prod - 1.0))
                
                if Residual < 1e-6:
                    converged = True
                    break
            
            if converged:
                best_rank = R
                best_result = (U, V, W)
                break
        
        if best_rank is not None:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
