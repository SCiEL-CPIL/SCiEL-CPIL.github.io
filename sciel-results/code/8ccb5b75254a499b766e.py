import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook initialization
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    target_T = np.ones((n, m, p))
    
    def check_decomposition(U_mat, V_mat, W_mat, tol=1e-6):
        U_reshaped = U_mat.reshape(n, m, -1)
        V_reshaped = V_mat.reshape(m, p, -1)
        W_reshaped = W_mat.reshape(p, n, -1)
        T_computed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T_computed, target_T, atol=tol)
    
    def check_decomposition_exact(U_mat, V_mat, W_mat):
        """Check with tighter tolerance for final verification."""
        U_reshaped = U_mat.reshape(n, m, -1)
        V_reshaped = V_mat.reshape(m, p, -1)
        W_reshaped = W_mat.reshape(p, n, -1)
        T_computed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T_computed, target_T, atol=1e-9)
    
    # Initial check
    if not check_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def try_reduce_rank(U_init, V_init, W_init, target_rank, max_iters=50):
        """
        Try to find a rank-reduced decomposition using iterative refinement.
        Strategy: Start from a valid rank-60 solution, zero out (60-target_rank) columns,
        then solve the remaining system using Gauss-Seidel-like iteration on the 
        bilinear constraints.
        """
        R0 = U_init.shape[1]
        if target_rank >= R0:
            return None
        
        # Create a copy and zero out some columns
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Zero out the last (R0 - target_rank) columns
        cols_to_zero = range(target_rank, R0)
        U[:, cols_to_zero] = 0
        V[:, cols_to_zero] = 0
        W[:, cols_to_zero] = 0
        
        # Now we have a rank-target_rank structure but it's not valid.
        # We need to adjust the remaining columns to satisfy the equations.
        # The equations are: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        
        # This is a system of n*m*p equations in R*(n*m + m*p + p*n) variables.
        # We can use a gradient descent approach on the residual.
        
        # Residual: resid[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1
        # We want to minimize sum of squares of resid.
        
        # Use simple gradient descent with line search
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            U_r = U.reshape(n, m, -1)
            V_r = V.reshape(m, p, -1)
            W_r = W.reshape(p, n, -1)
            
            # Compute residual
            T_comp = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            resid = T_comp - target_T
            
            # Compute gradients (approximate)
            # d(resid)/dU[i,j,r] = V[j,k,r] * W[k,i,r] for each k
            # Sum over k: grad_U[i,j,r] = sum_k resid[i,j,k] * V[j,k,r] * W[k,i,r]
            # This is expensive. Let's use a simpler approach.
            
            # Instead, use a coordinate descent / Gauss-Seidel approach:
            # For each column r, try to update U[:,r], V[:,r], W[:,r] to reduce residual.
            
            # Actually, let's try a different approach: 
            # Treat this as a least squares problem. For fixed V and W, 
            # the equations are linear in U. For fixed U and W, linear in V. Etc.
            
            # Block coordinate descent:
            # 1. Fix V, W. Solve for U: 
            #    For each (i,j): sum_r U[i,j,r] * (V[j,:,r] * W[:,i,r]).sum(axis=0) ... no
            #    Actually: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            #    For fixed (i,j), this is: sum_k T[i,j,k] * something...
            #    Let's define for each (i,j): 
            #    T[i,j,:] = sum_r U[i,j,r] * (V[j,:,r] * W[:,i,r])
            #    This is: T[i,j,:] = sum_r U[i,j,r] * vec_r where vec_r = V[j,:,r] * W[:,i,r]
            #    So for each (i,j), we have p equations: T[i,j,:] = U[i,j,:] @ M_ij
            #    where M_ij[r,k] = V[j,k,r] * W[k,i,r]
            #    This is a linear system: M_ij^T @ U[i,j,:] = T[i,j,:]
            #    (M_ij is p x R, U[i,j,:] is R x 1, result is p x 1)
            
            # Let's implement this block coordinate descent.
            converged = True
            for r in range(target_rank):
                if time.time() > deadline - 10:
                    return None
                
                # Update U column r
                # For each (i,j): T[i,j,:] = sum_s U[i,j,s] * V[j,:,s] * W[:,i,s]
                # Fix all U[i,j,s] for s != r. Then:
                # T[i,j,:] - sum_{s!=r} U[i,j,s] * V[j,:,s] * W[:,i,s] = U[i,j,r] * V[j,:,r] * W[:,i,s]
                # Let rhs[i,j,:] = T[i,j,:] - sum_{s!=r, s!=r} U[i,j,s] * V[j,:,s] * W[:,i,s]
                # Then rhs[i,j,:] = U[i,j,r] * (V[j,:,r] * W[:,i,r])
                # This is p equations in 1 unknown U[i,j,r].
                # rhs[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
                # So U[i,j,r] = rhs[i,j,k] / (V[j,k,r] * W[k,i,r]) if the denominator is nonzero.
                # This is overdetermined (p equations, 1 unknown). Use least squares.
                
                # Compute current T
                U_r = U.reshape(n, m, -1)
                V_r = V.reshape(m, p, -1)
                W_r = W.reshape(p, n, -1)
                T_comp = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                for i in range(n):
                    for j in range(m):
                        # Compute contribution from all columns except r
                        T_other = np.zeros(p)
                        for s in range(target_rank):
                            if s == r:
                                continue
                            T_other += U[i*m+j, s] * V[j*p + np.arange(p), s] * W[np.arange(p)*n + i, s]
                        
                        rhs = target_T[i, j, :] - T_other
                        # rhs[k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
                        denom = V[j*p + np.arange(p), r] * W[np.arange(p)*n + i, r]
                        
                        # Least squares: minimize sum_k (rhs[k] - U[i,j,r]*denom[k])^2
                        # Solution: U[i,j,r] = (rhs . denom) / (denom . denom)
                        dd = np.dot(denom, denom)
                        if abs(dd) < 1e-12:
                            continue
                        U_new = np.dot(rhs, denom) / dd
                        if abs(U_new - U[i*m+j, r]) > 1e-10:
                            U[i*m+j, r] = U_new
                            converged = False
                
                if time.time() > deadline - 10:
                    return None
                
                # Update V column r
                U_r = U.reshape(n, m, -1)
                V_r = V.reshape(m, p, -1)
                W_r = W.reshape(p, n, -1)
                T_comp = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                for j in range(m):
                    for k in range(p):
                        T_other = np.zeros(n)
                        for s in range(target_rank):
                            if s == r:
                                continue
                            T_other += U[np.arange(n)*m + j, s] * V[j*p + k, s] * W[k*n + np.arange(n), s]
                        
                        rhs = target_T[:, j, k] - T_other
                        # rhs[i] = V[j,k,r] * U[i,j,r] * W[k,i,r]
                        denom = U[np.arange(n)*m + j, r] * W[k*n + np.arange(n), r]
                        
                        dd = np.dot(denom, denom)
                        if abs(dd) < 1e-12:
                            continue
                        V_new = np.dot(rhs, denom) / dd
                        if abs(V_new - V[j*p + k, r]) > 1e-10:
                            V[j*p + k, r] = V_new
                            converged = False
                
                if time.time() > deadline - 10:
                    return None
                
                # Update W column r
                U_r = U.reshape(n, m, -1)
                V_r = V.reshape(m, p, -1)
                W_r = W.reshape(p, n, -1)
                T_comp = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                for k in range(p):
                    for i in range(n):
                        T_other = np.zeros(m)
                        for s in range(target_rank):
                            if s == r:
                                continue
                            T_other += U[i*m + np.arange(m), s] * V[np.arange(m)*p + k, s] * W[k*n + i, s]
                        
                        rhs = target_T[i, :, k] - T_other
                        # rhs[j] = W[k,i,r] * U[i,j,r] * V[j,k,r]
                        denom = U[i*m + np.arange(m), r] * V[np.arange(m)*p + k, r]
                        
                        dd = np.dot(denom, denom)
                        if abs(dd) < 1e-12:
                            continue
                        W_new = np.dot(rhs, denom) / dd
                        if abs(W_new - W[k*n + i, r]) > 1e-10:
                            W[k*n + i, r] = W_new
                            converged = False
            
            if converged:
                break
        
        # Check if we have a valid decomposition
        if check_decomposition(U, V, W, tol=1e-6):
            return U, V, W
        return None
    
    # Try to reduce rank from 59 down to some minimum
    for target_rank in range(rank - 1, 20, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_reduce_rank(U, V, W, target_rank, max_iters=30)
        if result is not None:
            U_red, V_red, W_red = result
            # Verify with tighter tolerance
            if check_decomposition_exact(U_red, V_red, W_red):
                best_rank = target_rank
                best_U, best_V, best_W = U_red, V_red, W_red
                # Update U, V, W for further reduction attempts
                U = U_red.copy()
                V = V_red.copy()
                W = W_red.copy()
            else:
                # Try to round to half-integers and check
                U_rounded = np.round(U_red * 2) / 2
                V_rounded = np.round(V_red * 2) / 2
                W_rounded = np.round(W_red * 2) / 2
                if check_decomposition_exact(U_rounded, V_rounded, W_rounded):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    U = U_rounded.copy()
                    V = V_rounded.copy()
                    W = W_rounded.copy()
                else:
                    break
    
    # Final verification
    if not check_decomposition(best_U, best_V, best_W, tol=1e-6):
        # Fall back to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to round to half-integers for the best solution
    best_U_rounded = np.round(best_U * 2) / 2
    best_V_rounded = np.round(best_V * 2) / 2
    best_W_rounded = np.round(best_W * 2) / 2
    
    if check_decomposition_exact(best_U_rounded, best_V_rounded, best_W_rounded):
        best_U, best_V, best_W = best_U_rounded, best_V_rounded, best_W_rounded
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
