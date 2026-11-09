import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Schoolbook solution (Rank 72)
    rank_school = n * m * p
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
    
    def check_decomposition(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        # Convert to numpy for speed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # We need to check all triples (i, j, k)
        # Reshape U to (n, m, R)
        U_reshaped = U_np.reshape(n, m, -1)
        # Reshape V to (m, p, R)
        V_reshaped = V_np.reshape(m, p, -1)
        # Reshape W to (p, n, R)
        W_reshaped = W_np.reshape(p, n, -1)
        
        # For each i, j, k, compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a batched dot product
        # U[i,j,:] dot V[j,k,:] dot W[k,i,:] ? No, it's element-wise product then sum.
        # U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        # Iterate over i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = U_reshaped[i, j, :]
                    v_vec = V_reshaped[j, k, :]
                    w_vec = W_reshaped[k, i, :]
                    prod = u_vec * v_vec * w_vec
                    s = np.sum(prod)
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def snap_to_half_integer(mat):
        # Round to nearest 0.5
        return [[int(round(x * 2)) / 2.0 for x in row] for row in mat]

    def try_rank(R):
        if time.time() > deadline:
            return None
            
        # Strategy: Use structured initialization based on schoolbook,
        # then attempt to merge columns via smooth optimization.
        # 
        # The key insight: if we can find two schoolbook columns that can be
        # "merged" with a correction term, we might reduce rank by 1.
        # 
        # For rank R = n*m*p - 1, we need to find coefficients such that
        # the tensor can be expressed with R terms instead of R+1.
        
        # Approach 1: Try to find a rank-R decomposition by perturbing schoolbook
        # and using smooth optimization to satisfy the constraints.
        
        # Number of constraints: n*m*p = 72
        # Number of variables: R*(n*m + m*p + p*n) = R*(12+24+18) = R*54
        # For R=71: 71*54 = 3834 variables, 72 constraints -> very underdetermined
        
        # Use a smooth objective: minimize sum of squared errors
        # with L2 regularization to keep entries small
        
        max_restarts = 30
        best_solution = None
        best_error = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize near schoolbook but with one column merged
            # Take first R-1 schoolbook columns, then create one merged column
            
            U_init = np.zeros((n*m, R))
            V_init = np.zeros((m*p, R))
            W_init = np.zeros((p*n, R))
            
            # Copy first R-1 columns from schoolbook
            for c in range(R-1):
                i = c // (m*p)
                j = (c // p) % m
                k = c % p
                U_init[i*m+j, c] = 1
                V_init[j*p+k, c] = 1
                W_init[k*n+i, c] = 1
            
            # Last column: random small values
            U_init[:, R-1] = np.random.uniform(-0.5, 0.5, n*m)
            V_init[:, R-1] = np.random.uniform(-0.5, 0.5, m*p)
            W_init[:, R-1] = np.random.uniform(-0.5, 0.5, p*n)
            
            # Smooth optimization using gradient descent
            # Objective: sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
            #            + lambda * (||U||^2 + ||V||^2 + ||W||^2)
            
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            lambda_reg = 0.01
            lr = 0.05
            max_iter = 200
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Compute current values
                U_r = U.reshape(n, m, R)
                V_r = V.reshape(m, p, R)
                W_r = W.reshape(p, n, R)
                
                # Compute error for each (i,j,k)
                errors = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            errors[i,j,k] = np.sum(U_r[i,j,:] * V_r[j,k,:] * W_r[k,i,:]) - 1.0
                
                # Compute gradients
                # d/dU[i,j,c] = 2 * sum_{i',j',k'} errors[i',j',k'] * delta(i',i)*delta(j',j)*V[j',k',c]*W[k',i',c]
                # This is expensive, so use finite differences or approximate
                
                # Simpler: use coordinate descent with random updates
                # Pick random entry and update
                
                # For efficiency, use a simple gradient approximation
                # Update U
                grad_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        for c in range(R):
                            # Gradient contribution from all (i,j,k)
                            g = 0
                            for k in range(p):
                                # error[i,j,k] * V[j,k,c] * W[k,i,c]
                                g += errors[i,j,k] * V[j,k,c] * W[k,i,c]
                            grad_U[i*m+j, c] = 2 * g + 2 * lambda_reg * U[i*m+j, c]
                
                # Update V
                grad_V = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        for c in range(R):
                            g = 0
                            for i in range(n):
                                g += errors[i,j,k] * U[i*m+j, c] * W[k*n+i, c]
                            grad_V[j*p+k, c] = 2 * g + 2 * lambda_reg * V[j*p+k, c]
                
                # Update W
                grad_W = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        for c in range(R):
                            g = 0
                            for j in range(m):
                                g += errors[i,j,k] * U[i*m+j, c] * V[j*p+k, c]
                            grad_W[k*n+i, c] = 2 * g + 2 * lambda_reg * W[k*n+i, c]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if it % 10 == 0:
                    err = np.sum(errors**2)
                    if err < best_error:
                        best_error = err
                        if err < 1e-8:
                            # Snap and check
                            U_snap = snap_to_half_integer(U.tolist())
                            V_snap = snap_to_half_integer(V.tolist())
                            W_snap = snap_to_half_integer(W.tolist())
                            if check_decomposition(U_snap, V_snap, W_snap):
                                best_solution = (U_snap, V_snap, W_snap)
                                return best_solution
                            else:
                                # Keep unsnapped as best
                                best_solution = (U.tolist(), V.tolist(), W.tolist())
                    
                    if err < 1e-10:
                        break
                
                if time.time() > deadline - 5:
                    break
            
            # Final check
            if best_solution is None and best_error < 1e-6:
                U_snap = snap_to_half_integer(U.tolist())
                V_snap = snap_to_half_integer(V.tolist())
                W_snap = snap_to_half_integer(W.tolist())
                if check_decomposition(U_snap, V_snap, W_snap):
                    best_solution = (U_snap, V_snap, W_snap)
                    return best_solution
                elif best_error < 1e-8:
                    best_solution = (U.tolist(), V.tolist(), W.tolist())
        
        return best_solution
    
    # Try to find rank 71 solution
    result = try_rank(71)
    
    if result is not None:
        U, V, W = result
        # Final verification
        if check_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
